/*
 * Copyright (c) ByteDance Ltd. and/or its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#include <folly/ScopeGuard.h>
#include "bolt/common/base/tests/GTestUtils.h"
#include "bolt/common/file/FileSystems.h"
#include "bolt/exec/SpillFile.h"
#include "bolt/exec/tests/utils/TempDirectoryPath.h"
#include "bolt/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "bolt/serializers/ArrowSerializer.h"
#include "bolt/serializers/PrestoSerializer.h"
#include "bolt/serializers/SpillSerializer.h"
#include "bolt/vector/fuzzer/VectorFuzzer.h"
#include "bolt/vector/tests/utils/VectorTestBase.h"

using namespace bytedance::bolt;
using namespace bytedance::bolt::test;
using bytedance::bolt::serializer::SpillVectorSerde;
using bytedance::bolt::serializer::presto::PrestoVectorSerde;

namespace {
// Models SpillInputStream's single reusable range, including its inherited
// nextView behavior. Random-access methods fail to catch accidental use.
class SequentialInput : public ByteInputStream {
 public:
  SequentialInput(const std::string& bytes, size_t chunkSize) {
    for (size_t offset = 0; offset < bytes.size(); offset += chunkSize) {
      segments_.push_back(
          {reinterpret_cast<uint8_t*>(const_cast<char*>(bytes.data())) + offset,
           static_cast<int32_t>(std::min(chunkSize, bytes.size() - offset)),
           0});
    }
    setRange(segments_.at(0));
  }
  bool atEnd() const override {
    return index_ + 1 == segments_.size() &&
        current_->position == current_->size;
  }
  size_t size() const override {
    BOLT_FAIL("Sequential input has no size");
  }
  size_t remainingSize() const override {
    BOLT_FAIL("Sequential input has no remainingSize");
  }
  std::streampos tellp() const override {
    BOLT_FAIL("Sequential input cannot tell");
  }
  void seekp(std::streampos) override {
    BOLT_FAIL("Sequential input cannot seek");
  }

 private:
  void next(bool throwIfPastEnd = true) override {
    if (index_ + 1 == segments_.size()) {
      BOLT_CHECK(!throwIfPastEnd, "Truncated sequential input");
      return;
    }
    setRange(segments_[++index_]);
  }
  std::vector<ByteRange> segments_;
  size_t index_{0};
};
} // namespace

class SpillSerializerTest : public testing::Test, public VectorTestBase {
 protected:
  using Options = VectorSerde::Options;
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
    filesystems::registerLocalFileSystem();
  }

  std::unique_ptr<ByteInputStream> input(const folly::IOBuf& buffer) {
    std::vector<ByteRange> ranges;
    for (auto range : buffer) {
      size_t offset = 0;
      while (offset < range.size()) {
        const auto size = std::min<size_t>(64 << 20, range.size() - offset);
        ranges.push_back(
            {const_cast<uint8_t*>(range.data()) + offset,
             static_cast<int32_t>(size),
             0});
        offset += size;
      }
    }
    return std::make_unique<ByteInputStream>(std::move(ranges));
  }

  std::unique_ptr<folly::IOBuf> encode(
      const RowVectorPtr& row,
      common::CompressionKind codec,
      bool checksum = false,
      VectorSerde* serde = SpillVectorSerde::get()) {
    Options options(true, codec);
    StreamArena arena(pool_.get());
    auto serializer = serde->createSerializer(
        asRowType(row->type()), row->size(), &arena, &options);
    serializer->append(row);
    serializer::presto::PrestoOutputStreamListener listener;
    const auto estimate = serializer->maxSerializedSize();
    IOBufOutputStream output(
        *pool_,
        checksum ? &listener : nullptr,
        std::min<size_t>(64 << 20, std::max<size_t>(64 << 10, estimate)));
    serializer->flush(&output);
    if (codec == common::CompressionKind_NONE) {
      EXPECT_EQ(static_cast<uint64_t>(output.tellp()), estimate);
    } else {
      EXPECT_LE(static_cast<uint64_t>(output.tellp()), estimate);
    }
    return output.getIOBuf();
  }

  RowVectorPtr decode(
      const folly::IOBuf& buffer,
      const RowTypePtr& type,
      common::CompressionKind codec) {
    auto stream = input(buffer);
    Options options(true, codec);
    RowVectorPtr result;
    SpillVectorSerde::get()->deserialize(
        stream.get(), pool_.get(), type, &result, &options);
    EXPECT_TRUE(stream->atEnd());
    return result;
  }

  void spillRoundTrip(
      const RowVectorPtr& row,
      common::CompressionKind compression,
      std::optional<VectorSerde::Kind> kind,
      bool direct,
      bool multipleWrites = false) {
    PrestoVectorSerde::registerVectorSerde();
    PrestoVectorSerde::registerNamedVectorSerde();
    serializer::arrowserde::ArrowVectorSerde::registerNamedVectorSerde();
    auto cleanup = folly::makeGuard([] {
      if (isRegisteredVectorSerde()) {
        deregisterVectorSerde();
      }
      if (isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
        deregisterNamedVectorSerde(VectorSerde::Kind::kPresto);
      }
      deregisterNamedVectorSerde(VectorSerde::Kind::kArrow);
    });
    auto directory = exec::test::TempDirectoryPath::create();
    common::SpillConfig::SpillIOConfig config{
        [&]() -> const std::string& { return directory->path; },
        [](uint64_t) {},
        "spill",
        0,
        false,
        1 << 20,
        compression,
        "",
        kind};
    folly::Synchronized<common::SpillStats> stats;
    exec::SpillWriter writer(
        asRowType(row->type()),
        {},
        directory->path + "/spill",
        0,
        config,
        pool_.get(),
        &stats);
    IndexRange range{0, row->size()};
    if (multipleWrites) {
      const vector_size_t half = row->size() / 2;
      IndexRange first{0, half};
      IndexRange second{half, row->size() - half};
      EXPECT_EQ(writer.write(row, folly::Range(&first, 1)), 0);
      EXPECT_EQ(writer.write(row, folly::Range(&second, 1)), 0);
    } else if (direct) {
      writer.writeAndFlush(row, folly::Range(&range, 1));
    } else {
      writer.write(row, folly::Range(&range, 1));
    }
    auto files = writer.finish();
    ASSERT_EQ(files.size(), 1);
    if (kind != VectorSerde::Kind::kArrow) {
      EXPECT_EQ(files[0].serdeKind, VectorSerde::Kind::kSpill);
      auto file = filesystems::getFileSystem(files[0].path, nullptr)
                      ->openFileForRead(files[0].path);
      uint32_t magic;
      file->pread(0, sizeof(magic), &magic);
      EXPECT_EQ(magic, 0xB0175F01u);
      // The file must not depend on the lifetime or identity of either
      // Presto registration after the writer has fixed its actual format.
      deregisterVectorSerde();
      deregisterNamedVectorSerde(VectorSerde::Kind::kPresto);
    } else {
      EXPECT_EQ(files[0].serdeKind, VectorSerde::Kind::kArrow);
    }
    auto reader = exec::SpillReadFile::create(files[0], pool_.get(), false);
    RowVectorPtr result;
    ASSERT_TRUE(reader->nextBatch(result));
    assertEqualVectors(row, result);
    EXPECT_FALSE(reader->nextBatch(result));
  }

  static bool runLargePages() {
    const auto* value = std::getenv("BOLT_RUN_LARGE_PAGE_TESTS");
    return value != nullptr && std::string_view(value) == "1";
  }
};

TEST_F(SpillSerializerTest, smallSpillUsesInternalFormat) {
  auto row =
      makeRowVector({makeFlatVector<Timestamp>({Timestamp(12, 123456789)})});
  for (auto kind :
       {std::optional<VectorSerde::Kind>{},
        std::optional{VectorSerde::Kind::kPresto},
        std::optional{VectorSerde::Kind::kSpill},
        std::optional{VectorSerde::Kind::kArrow}}) {
    for (auto codec :
         {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
      for (bool direct : {false, true}) {
        spillRoundTrip(row, codec, kind, direct);
      }
    }
  }
}

TEST_F(SpillSerializerTest, independentBooleanRows) {
  // Run this test alone as well: the internal API must work before any
  // Presto registration has initialized the Boolean expansion table.
  ASSERT_FALSE(isRegisteredVectorSerde());
  auto row = makeRowVector({makeNullableFlatVector<bool>(
      {true, false, std::nullopt, true, false, true, false, true, false})});
  const vector_size_t rows[] = {8, 0, 2, 1, 5, 3, 7};
  auto expected = makeRowVector({makeNullableFlatVector<bool>(
      {false, true, std::nullopt, false, true, true, true})});
  StreamArena arena(pool_.get());
  auto serializer = SpillVectorSerde::get()->createSerializer(
      asRowType(row->type()), 7, &arena);
  Scratch scratch;
  serializer->append(row, folly::Range(rows, 7), scratch);
  IOBufOutputStream output(*pool_);
  serializer->flush(&output);
  auto buffer = output.getIOBuf();
  assertEqualVectors(
      expected,
      decode(*buffer, asRowType(row->type()), common::CompressionKind_NONE));
  serializer->clear();
  serializer->append(expected);
  IOBufOutputStream second(*pool_);
  serializer->flush(&second);
  auto next = second.getIOBuf();
  assertEqualVectors(
      expected,
      decode(*next, asRowType(row->type()), common::CompressionKind_NONE));
}

TEST_F(SpillSerializerTest, sizeQueryBeforeAppend) {
  auto values =
      makeNullableFlatVector<int64_t>({std::nullopt, 7, 9, std::nullopt});
  auto nested = makeRowVector({values});
  nested->setNull(1, true);
  auto row = makeRowVector({values, nested});
  for (auto codec :
       {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
    spillRoundTrip(row, codec, std::nullopt, false, true);
    StreamArena arena(pool_.get());
    Options options(true, codec);
    auto serializer = SpillVectorSerde::get()->createSerializer(
        asRowType(row->type()), row->size(), &arena, &options);
    Scratch scratch;
    const IndexRange first{0, 2};
    const IndexRange second{2, 2};
    serializer->append(row, folly::Range(&first, 1), scratch);
    EXPECT_GT(serializer->maxSerializedSize(), 0);
    serializer->append(row, folly::Range(&second, 1), scratch);
    const auto size = serializer->maxSerializedSize();
    IOBufOutputStream output(*pool_);
    serializer->flush(&output);
    EXPECT_LE(static_cast<size_t>(output.tellp()), size);
    auto bytes = output.getIOBuf();
    assertEqualVectors(row, decode(*bytes, asRowType(row->type()), codec));
  }
}

TEST_F(SpillSerializerTest, rowCountLimit) {
  const auto limit = std::numeric_limits<vector_size_t>::max();
  auto row = std::make_shared<RowVector>(
      pool_.get(), ROW({}, {}), nullptr, limit, std::vector<VectorPtr>{});
  StreamArena arena(pool_.get());
  auto serializer = SpillVectorSerde::get()->createSerializer(
      asRowType(row->type()), 1, &arena);
  Scratch scratch;
  const IndexRange all{0, limit};
  serializer->append(row, folly::Range(&all, 1), scratch);
  const IndexRange one{0, 1};
  BOLT_ASSERT_THROW(
      serializer->append(row, folly::Range(&one, 1), scratch),
      "Spill row count exceeds INT32_MAX");
  const vector_size_t id = 0;
  BOLT_ASSERT_THROW(
      serializer->append(row, folly::Range(&id, 1), scratch),
      "Spill row count exceeds INT32_MAX");
}

TEST_F(SpillSerializerTest, prestoGoldenBytes) {
  PrestoVectorSerde presto;
  auto row = makeRowVector({makeFlatVector<int64_t>({7, 9})});
  const unsigned char header[] = {2, 0, 0, 0, 0, 39, 0, 0, 0, 39, 0, 0, 0};
  const unsigned char payload[] = {
      1,   0,   0,   0,   10,  0, 0, 0, 'L', 'O', 'N', 'G', '_',
      'A', 'R', 'R', 'A', 'Y', 2, 0, 0, 0,   0,   7,   0,   0,
      0,   0,   0,   0,   0,   9, 0, 0, 0,   0,   0,   0,   0};
  std::string expected(reinterpret_cast<const char*>(header), sizeof(header));
#ifdef BOLT_ENABLE_CRC
  expected.append(8, '\0');
#endif
  expected.append(reinterpret_cast<const char*>(payload), sizeof(payload));
  auto bytes = encode(row, common::CompressionKind_NONE, false, &presto);
  EXPECT_EQ(bytes->moveToFbString().toStdString(), expected);
}

TEST_F(SpillSerializerTest, mixedTypesAndEncodings) {
  const auto type = ROW(
      {BOOLEAN(),
       TINYINT(),
       SMALLINT(),
       INTEGER(),
       BIGINT(),
       HUGEINT(),
       REAL(),
       DOUBLE(),
       VARCHAR(),
       VARBINARY(),
       TIMESTAMP(),
       ARRAY(BIGINT()),
       MAP(VARCHAR(), ARRAY(DOUBLE())),
       ROW({INTEGER(), VARCHAR()})});
  VectorFuzzer::Options options;
  options.vectorSize = 101;
  options.nullRatio = 0.2;
  options.containerLength = 5;
  for (uint32_t seed : {13, 27, 42}) {
    VectorFuzzer fuzzer(options, pool_.get(), seed);
    auto row = fuzzer.fuzzRow(type);
    // A page is a table of columns, not a nullable top-level ROW value.
    // Keep nulls in every child (including nested rows).
    row->resetNulls();
    for (auto codec :
         {common::CompressionKind_NONE,
          common::CompressionKind_LZ4,
          common::CompressionKind_ZSTD,
          common::CompressionKind_SNAPPY,
          common::CompressionKind_ZLIB,
          common::CompressionKind_GZIP}) {
      SCOPED_TRACE(fmt::format("seed={} codec={}", seed, codec));
      auto buffer = encode(row, codec, true);
      assertEqualVectors(row, decode(*buffer, type, codec));
      Options serdeOptions(true, codec);
      auto batch = SpillVectorSerde::get()->createBatchSerializer(
          pool_.get(), &serdeOptions);
      IOBufOutputStream output(*pool_);
      batch->serialize(row, &output);
      auto bytes = output.getIOBuf();
      assertEqualVectors(row, decode(*bytes, type, codec));
    }
  }
}

TEST_F(SpillSerializerTest, specializedTypes) {
  auto timestamp = makeFlatVector<int64_t>({10000, -10000, 0});
  auto timezone = makeFlatVector<int16_t>({0, 12, 36});
  auto zoned = std::make_shared<RowVector>(
      pool_.get(),
      TIMESTAMP_WITH_TIME_ZONE(),
      nullptr,
      3,
      std::vector<VectorPtr>{timestamp, timezone});
  zoned->setNull(1, true);
  auto decimal = makeFlatVector<int128_t>(
      {DecimalUtil::kLongDecimalMin, 0, DecimalUtil::kLongDecimalMax},
      DECIMAL(38, 5));
  auto row = makeRowVector(
      {decimal,
       makeFlatVector<int64_t>({-12345, 0, 23456}, INTERVAL_DAY_TIME()),
       zoned,
       BaseVector::createNullConstant(UNKNOWN(), 3, pool_.get()),
       makeLazyFlatVector<int64_t>(3, [](auto i) { return i * 13; }),
       makeArrayOfRowVector(ROW({UNKNOWN()}), {{}, {}, {}})});
  for (auto codec :
       {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
    auto bytes = encode(row, codec, true);
    assertEqualVectors(row, decode(*bytes, asRowType(row->type()), codec));
    spillRoundTrip(row, codec, std::nullopt, false, true);
  }
}

TEST_F(SpillSerializerTest, sequentialPagesAndAppend) {
  auto first =
      makeRowVector({makeNullableFlatVector<int64_t>({7, std::nullopt, 9})});
  auto second =
      makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt, 11})});
  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      {7, std::nullopt, 9, std::nullopt, 11})});
  for (auto codec :
       {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
    auto a = encode(first, codec, true)->moveToFbString().toStdString();
    auto b = encode(second, codec, true)->moveToFbString().toStdString();
    const auto bytes = a + b + a;
    Options options(true, codec);
    for (size_t chunk : {1, 3, 7, 33, 64}) {
      SequentialInput stream(bytes, chunk);
      RowVectorPtr result;
      SpillVectorSerde::get()->deserialize(
          &stream, pool_.get(), asRowType(first->type()), &result, &options);
      SpillVectorSerde::get()->deserialize(
          &stream, pool_.get(), asRowType(first->type()), &result, 3, &options);
      assertEqualVectors(expected, result);
      // Reuse the destination after a non-byte-aligned append.
      SpillVectorSerde::get()->deserialize(
          &stream, pool_.get(), asRowType(first->type()), &result, &options);
      assertEqualVectors(first, result);
      EXPECT_TRUE(stream.atEnd());
      SequentialInput projected(bytes, chunk);
      RowVectorPtr empty;
      SpillVectorSerde::get()->deserialize(
          &projected, pool_.get(), ROW({}, {}), &empty, &options);
      EXPECT_EQ(empty->size(), 3);
      result.reset();
      SpillVectorSerde::get()->deserialize(
          &projected,
          pool_.get(),
          asRowType(second->type()),
          &result,
          &options);
      assertEqualVectors(second, result);
    }
  }
}

TEST_F(SpillSerializerTest, checksumOracleAndCorruption) {
  auto row = makeRowVector({makeFlatVector<int64_t>({7, 9})});
  auto bytes =
      encode(row, common::CompressionKind_NONE)->moveToFbString().toStdString();
  // Add a checksum independently from production helpers. CRC covers the
  // payload followed by the first 25 header bytes, including both int64 sizes.
  ASSERT_EQ(bytes.size(), 33 + 39);
  bytes[8] = 4;
  bits::Crc32 crc;
  crc.process_bytes(bytes.data() + 33, bytes.size() - 33);
  crc.process_bytes(bytes.data(), 25);
  const uint64_t checksum = crc.checksum();
  std::memcpy(bytes.data() + 25, &checksum, sizeof(checksum));
  for (bool corrupt : {false, true}) {
    auto data = bytes;
    if (corrupt) {
      data.back() ^= 1;
    }
    SequentialInput stream(data, 3);
    RowVectorPtr result;
    if (corrupt) {
      BOLT_ASSERT_THROW(
          SpillVectorSerde::get()->deserialize(
              &stream, pool_.get(), asRowType(row->type()), &result),
          "Corrupted spill page");
    } else {
      SpillVectorSerde::get()->deserialize(
          &stream, pool_.get(), asRowType(row->type()), &result);
      assertEqualVectors(row, result);
      EXPECT_TRUE(stream.atEnd());
    }
  }
#ifdef BOLT_ENABLE_CRC
  auto actual = encode(row, common::CompressionKind_NONE, true)
                    ->moveToFbString()
                    .toStdString();
  EXPECT_EQ(actual, bytes);
#endif
}

TEST_F(SpillSerializerTest, invalidHeadersAndTruncation) {
  auto row = makeRowVector({makeFlatVector<int64_t>({7, 9})});
  const auto good =
      encode(row, common::CompressionKind_NONE)->moveToFbString().toStdString();
  std::vector<std::string> invalid;
  for (auto offset : {0, 8}) {
    auto data = good;
    data[offset] ^= 0x40;
    invalid.push_back(data);
  }
  for (auto offset : {9, 17}) {
    for (int64_t value : {-1, 0, 1, 38, 40}) {
      auto data = good;
      std::memcpy(data.data() + offset, &value, sizeof(value));
      invalid.push_back(data);
    }
  }
  auto extra = good + "x";
  int64_t length = 40;
  std::memcpy(extra.data() + 9, &length, 8);
  std::memcpy(extra.data() + 17, &length, 8);
  invalid.push_back(extra);
  invalid.push_back(good.substr(0, 32));
  invalid.push_back(good.substr(0, good.size() - 1));
  for (const auto& data : invalid) {
    SequentialInput stream(data, 3);
    RowVectorPtr result;
    EXPECT_THROW(
        SpillVectorSerde::get()->deserialize(
            &stream, pool_.get(), asRowType(row->type()), &result),
        BoltRuntimeError);
  }
  PrestoVectorSerde presto;
  auto legacy = encode(row, common::CompressionKind_NONE, false, &presto);
  BOLT_ASSERT_THROW(
      decode(*legacy, asRowType(row->type()), common::CompressionKind_NONE),
      "Unknown spill format/version");
}

TEST_F(SpillSerializerTest, fileRefill) {
  auto row = makeRowVector({makeFlatVector<std::string>(
      4096, [](auto i) { return std::string(1024, 'a' + i % 26); })});
  for (auto codec :
       {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
    spillRoundTrip(row, codec, std::nullopt, false);
    spillRoundTrip(row, codec, VectorSerde::Kind::kPresto, true);
  }
}

TEST_F(SpillSerializerTest, singleRowOverTwoGiB) {
  if (!runLargePages()) {
    GTEST_SKIP() << "Set BOLT_RUN_LARGE_PAGE_TESTS=1 (about 24 GiB RAM)";
  }
  for (const vector_size_t count : {16 * 1024 * 1024, 24 * 1024 * 1024}) {
    auto values = makeArrayVector<double>(
        1, [=](auto) { return count; }, [](auto i) { return i * 0.125; });
    auto counts = makeArrayVector<int64_t>(
        1, [=](auto) { return count; }, [](auto i) { return 1 + i % 17; });
    auto state = makeRowVector(
        {makeArrayVector<double>({{0.5}}),
         makeFlatVector<bool>({false}),
         values,
         counts});
    auto row = makeRowVector(std::vector<VectorPtr>(12, state));
    for (auto codec :
         {common::CompressionKind_NONE, common::CompressionKind_ZSTD}) {
      auto buffer = encode(row, codec, true);
      EXPECT_LT(buffer->countChainElements(), 20000);
      auto header = input(*buffer);
      EXPECT_EQ(header->read<uint32_t>(), 0xB0175F01u);
      EXPECT_EQ(header->read<int32_t>(), 1);
      header->read<uint8_t>();
      const auto size = header->read<int64_t>();
      EXPECT_GT(size, std::numeric_limits<int32_t>::max());
      if (count == 24 * 1024 * 1024) {
        EXPECT_GT(size, std::numeric_limits<uint32_t>::max());
      }
      EXPECT_EQ(decode(*buffer, ROW({}, {}), codec)->size(), 1);
      assertEqualVectors(row, decode(*buffer, asRowType(row->type()), codec));
      buffer.reset();
      if (count == 16 * 1024 * 1024) {
        spillRoundTrip(
            row, codec, std::nullopt, codec == common::CompressionKind_ZSTD);
      }
    }
    if (count == 16 * 1024 * 1024) {
      auto buffer = encode(row, common::CompressionKind_LZ4);
      auto header = input(*buffer);
      header->read<uint32_t>();
      header->read<int32_t>();
      EXPECT_EQ(header->read<uint8_t>() & 1, 0);
      assertEqualVectors(
          row,
          decode(*buffer, asRowType(row->type()), common::CompressionKind_LZ4));
    }
  }
}

TEST_F(SpillSerializerTest, compressedSizeOverTwoGiB) {
  if (!runLargePages()) {
    GTEST_SKIP() << "Set BOLT_RUN_LARGE_PAGE_TESTS=1 (about 24 GiB RAM)";
  }
  constexpr vector_size_t count = 16 * 1024 * 1024;
  auto randomBits = [](uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto values = makeArrayVector<double>(
      1,
      [](auto) { return count; },
      [&](auto i) {
        return static_cast<double>(randomBits(i) >> 11) * 0x1.0p-53;
      });
  auto counts = makeArrayVector<int64_t>(
      1,
      [](auto) { return count; },
      [&](auto i) {
        return 1 + (randomBits(i + count) & ((uint64_t{1} << 38) - 1));
      });
  auto state = makeRowVector(
      {makeArrayVector<double>({{0.5}}),
       makeFlatVector<bool>({false}),
       values,
       counts});
  auto row = makeRowVector(std::vector<VectorPtr>(12, state));
  auto buffer = encode(row, common::CompressionKind_ZSTD, true);
  auto header = input(*buffer);
  EXPECT_EQ(header->read<uint32_t>(), 0xB0175F01u);
  EXPECT_EQ(header->read<int32_t>(), 1);
  EXPECT_EQ(header->read<uint8_t>() & 1, 1);
  const auto uncompressed = header->read<int64_t>();
  const auto stored = header->read<int64_t>();
  RecordProperty("uncompressedBytes", std::to_string(uncompressed));
  RecordProperty("compressedBytes", std::to_string(stored));
  EXPECT_GT(stored, std::numeric_limits<int32_t>::max());
  EXPECT_LT(stored, uncompressed);
  assertEqualVectors(
      row,
      decode(*buffer, asRowType(row->type()), common::CompressionKind_ZSTD));
  buffer.reset();
  spillRoundTrip(row, common::CompressionKind_ZSTD, std::nullopt, true);
}

TEST_F(SpillSerializerTest, leafBufferLimit) {
  if (!runLargePages()) {
    GTEST_SKIP() << "Set BOLT_RUN_LARGE_PAGE_TESTS=1 (about 6 GiB RAM)";
  }
  {
    auto valid = makeRowVector({makeArrayVector<int64_t>(
        1, [](auto) { return (1 << 28) - 1; }, [](auto i) { return i; })});
    auto bytes = encode(valid, common::CompressionKind_NONE);
    assertEqualVectors(
        valid,
        decode(*bytes, asRowType(valid->type()), common::CompressionKind_NONE));
  }
  auto row = makeRowVector({makeArrayVector<int64_t>(
      1, [](auto) { return 1 << 28; }, [](auto i) { return i; })});
  BOLT_ASSERT_THROW(
      encode(row, common::CompressionKind_NONE),
      "Spill leaf values exceed INT32_MAX");
}
