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
#include "bolt/serializers/SpillSerializer.h"

#include <folly/compression/Zstd.h>
#include "bolt/common/base/Crc.h"
#include "bolt/common/flags/BoltFlags.h"
#include "bolt/common/memory/ByteStream.h"
#include "bolt/serializers/PrestoSerializer.h"
#include "bolt/serializers/PrestoSerializerInternal.h"

namespace bytedance::bolt::serializer {
namespace {
// Version 1. Unlike a standard Presto page, every spill page has this magic,
// two int64 lengths and a checksum slot, regardless of build configuration.
constexpr uint32_t kSpillMagic = 0xB0175F01;
constexpr uint8_t kCompressed = 1;
constexpr uint8_t kChecksum = 4;
constexpr int32_t kIoChunkSize = 64 * 1024 * 1024;
constexpr int64_t kHeaderSize = 4 + 4 + 1 + 8 + 8 + 8;

struct PageHeader {
  int32_t numRows;
  uint8_t flags;
  int64_t uncompressedSize;
  int64_t storedSize;
  uint64_t checksum{0};
};

template <typename T>
void writeField(OutputStream* out, T value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeHeader(OutputStream* out, const PageHeader& header) {
  writeField(out, kSpillMagic);
  writeField(out, header.numRows);
  writeField(out, header.flags);
  writeField(out, header.uncompressedSize);
  writeField(out, header.storedSize);
  writeField(out, header.checksum);
}

uint64_t finishChecksum(bits::Crc32 crc, const PageHeader& header) {
  crc.process_bytes(&kSpillMagic, sizeof(kSpillMagic));
  crc.process_bytes(&header.numRows, sizeof(header.numRows));
  crc.process_bytes(&header.flags, sizeof(header.flags));
  crc.process_bytes(&header.uncompressedSize, sizeof(header.uncompressedSize));
  crc.process_bytes(&header.storedSize, sizeof(header.storedSize));
  return crc.checksum();
}

void writeBuffer(OutputStream* out, const folly::IOBuf& buffer) {
  for (auto range : buffer) {
    size_t offset = 0;
    while (offset < range.size()) {
      const auto size = std::min<size_t>(kIoChunkSize, range.size() - offset);
      out->write(reinterpret_cast<const char*>(range.data() + offset), size);
      offset += size;
    }
  }
}

// Restricts column decoding to one page and computes CRC as buffers are
// borrowed. It never seeks or asks the underlying spill stream for its size.
// A borrowed view is consumed before the source can refill and invalidate it.
class PageInputStream : public ByteInputStream {
 public:
  PageInputStream(ByteInputStream* source, int64_t size, bool checksum)
      : source_(source), remaining_(size), checksum_(checksum) {
    setRange({nullptr, 0, 0});
  }

  bool atEnd() const override {
    return remaining_ == 0 && current_->position == current_->size;
  }

  size_t remainingSize() const override {
    return remaining_ + current_->size - current_->position;
  }

  std::string_view nextView(int32_t size) override {
    BOLT_CHECK_GE(size, 0);
    if (size == 0 || atEnd()) {
      return {};
    }
    if (current_->position == current_->size) {
      next();
    }
    return ByteInputStream::nextView(size);
  }

  void readBytes(uint8_t* data, int32_t size) override {
    BOLT_CHECK_GE(size, 0);
    BOLT_CHECK_LE(size, remainingSize(), "Reading past spill page");
    ByteInputStream::readBytes(data, size);
  }

  void skip(int32_t size) override {
    BOLT_CHECK_GE(size, 0);
    BOLT_CHECK_LE(size, remainingSize(), "Skipping past spill page");
    ByteInputStream::skip(size);
  }

  const bits::Crc32& crc() const {
    return crc_;
  }

 private:
  void next(bool throwIfPastEnd = true) override {
    if (remaining_ == 0) {
      BOLT_CHECK(!throwIfPastEnd, "Reading past spill page");
      return;
    }
    BOLT_CHECK(!source_->atEnd(), "Truncated spill page");
    auto view = source_->nextView(
        static_cast<int32_t>(std::min<int64_t>(remaining_, kIoChunkSize)));
    if (view.empty()) {
      // SpillInputStream has one reusable range. Its inherited nextView()
      // returns empty at the buffer boundary without calling next(). Reading
      // one byte triggers the refill; subsequent bytes can be borrowed again.
      firstByte_ = source_->readByte();
      view = std::string_view(reinterpret_cast<const char*>(&firstByte_), 1);
    }
    remaining_ -= view.size();
    if (checksum_) {
      crc_.process_bytes(view.data(), view.size());
    }
    setRange(
        {reinterpret_cast<uint8_t*>(const_cast<char*>(view.data())),
         static_cast<int32_t>(view.size()),
         0});
  }

  ByteInputStream* const source_;
  int64_t remaining_;
  const bool checksum_;
  uint8_t firstByte_{0};
  bits::Crc32 crc_;
};

class SpillVectorSerializer : public VectorSerializer {
 public:
  SpillVectorSerializer(
      const RowTypePtr& type,
      int32_t numRows,
      StreamArena* arena,
      const VectorSerde::Options& options)
      : arena_(arena),
        columns_(presto::detail::createColumnSerializer(
            type,
            numRows,
            arena,
            options.useLosslessTimestamp)),
        codec_(
            options.compressionKind == common::CompressionKind_ZSTD
                ? folly::io::zstd::getCodec(folly::io::zstd::Options(
                      FLAGS_bolt_shuffle_zstd_compression_level
                          ? FLAGS_bolt_shuffle_zstd_compression_level
                          : 3))
                : common::compressionKindToCodec(options.compressionKind)) {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& scratch) override {
    int64_t count = columns_->numRows();
    for (const auto& range : ranges) {
      BOLT_CHECK_GE(range.size, 0);
      count += range.size;
      BOLT_CHECK_LE(
          count,
          std::numeric_limits<int32_t>::max(),
          "Spill row count exceeds INT32_MAX");
    }
    columns_->append(vector, ranges, scratch);
  }

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const vector_size_t*>& rows,
      Scratch& scratch) override {
    BOLT_CHECK_LE(
        rows.size(),
        std::numeric_limits<int32_t>::max() - columns_->numRows(),
        "Spill row count exceeds INT32_MAX");
    columns_->append(vector, rows, scratch);
  }

  bool supportsAppendRows() const override {
    return true;
  }

  size_t maxSerializedSize() const override {
    const auto size = columns_->columnsSize();
    const auto stored =
        codec_->type() != folly::io::CodecType::NO_COMPRESSION &&
            size <= codec_->maxUncompressedLength()
        ? std::max<uint64_t>(size, codec_->maxCompressedLength(size))
        : size;
    BOLT_CHECK_LE(stored, std::numeric_limits<int64_t>::max() - kHeaderSize);
    return kHeaderSize + stored;
  }

  void flush(OutputStream* out) override {
    auto* listener =
        dynamic_cast<presto::PrestoOutputStreamListener*>(out->listener());
    PageHeader header{columns_->numRows(), 0, 0, 0};
#ifdef BOLT_ENABLE_CRC
    if (listener != nullptr) {
      header.flags |= kChecksum;
    }
#endif
    if (listener) {
      listener->reset();
      listener->pause();
    }

    std::unique_ptr<folly::IOBuf> buffer;
    if (codec_->type() != folly::io::CodecType::NO_COMPRESSION) {
      IOBufOutputStream payload(
          *arena_->pool(),
          nullptr,
          static_cast<int32_t>(std::min<size_t>(
              kIoChunkSize, std::max<size_t>(64 * 1024, arena_->size()))));
      columns_->flushColumns(&payload);
      header.uncompressedSize = payload.tellp();
      buffer = payload.getIOBuf();
      if (header.uncompressedSize <= codec_->maxUncompressedLength()) {
        auto compressed = codec_->compress(buffer.get());
        const auto size = compressed->computeChainDataLength();
        if (size < header.uncompressedSize) {
          buffer = std::move(compressed);
          header.flags |= kCompressed;
        }
      }
      header.storedSize = buffer->computeChainDataLength();
    }

    const auto start = out->tellp();
    writeHeader(out, header);
    if (listener && (header.flags & kChecksum)) {
      listener->resume();
    }
    if (buffer) {
      writeBuffer(out, *buffer);
    } else {
      // Fixed framing can be reserved before the size is known. Avoid a
      // second payload buffer or a counting pass on the uncompressed path.
      columns_->flushColumns(out);
    }
    if (listener) {
      listener->pause();
    }
    const auto end = out->tellp();
    const int64_t storedSize = end - start - kHeaderSize;
    if (buffer) {
      BOLT_CHECK_EQ(storedSize, header.storedSize);
    } else {
      header.uncompressedSize = header.storedSize = storedSize;
    }
    if (header.flags & kChecksum) {
      header.checksum = finishChecksum(listener->crc(), header);
    }
    out->seekp(start);
    writeHeader(out, header);
    out->seekp(end);
  }

  void clear() override {
    columns_->clear();
  }

 private:
  StreamArena* const arena_;
  const std::unique_ptr<presto::detail::ColumnSerializer> columns_;
  const std::unique_ptr<folly::io::Codec> codec_;
};
} // namespace

SpillVectorSerde* SpillVectorSerde::get() {
  static SpillVectorSerde serde;
  return &serde;
}

void SpillVectorSerde::estimateSerializedSize(
    VectorPtr vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  presto::PrestoVectorSerde().estimateSerializedSize(
      std::move(vector), ranges, sizes, scratch);
}

void SpillVectorSerde::estimateSerializedSize(
    VectorPtr vector,
    folly::Range<const vector_size_t*> rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  presto::PrestoVectorSerde().estimateSerializedSize(
      std::move(vector), rows, sizes, scratch);
}

std::unique_ptr<VectorSerializer> SpillVectorSerde::createSerializer(
    RowTypePtr type,
    int32_t numRows,
    StreamArena* arena,
    const Options* options) {
  return std::make_unique<SpillVectorSerializer>(
      type, numRows, arena, options ? *options : Options{});
}

void SpillVectorSerde::deserialize(
    ByteInputStream* source,
    memory::MemoryPool* pool,
    RowTypePtr type,
    RowVectorPtr* result,
    vector_size_t resultOffset,
    const Options* options) {
  BOLT_CHECK_EQ(
      source->read<uint32_t>(), kSpillMagic, "Unknown spill format/version");
  PageHeader header{
      source->read<int32_t>(),
      source->read<uint8_t>(),
      source->read<int64_t>(),
      source->read<int64_t>(),
      source->read<uint64_t>()};
  BOLT_CHECK_GE(header.numRows, 0);
  BOLT_CHECK_EQ(
      header.flags & ~(kCompressed | kChecksum), 0, "Invalid spill flags");
  BOLT_CHECK_GE(header.uncompressedSize, int64_t{sizeof(int32_t)});
  BOLT_CHECK_GT(header.storedSize, 0);
  if (header.flags & kCompressed) {
    BOLT_CHECK_LT(header.storedSize, header.uncompressedSize);
  } else {
    BOLT_CHECK_EQ(header.storedSize, header.uncompressedSize);
  }
  if (!(header.flags & kChecksum)) {
    BOLT_CHECK_EQ(header.checksum, 0);
  }
  const Options defaultOptions;
  const auto& serdeOptions = options ? *options : defaultOptions;
  PageInputStream payload(source, header.storedSize, header.flags & kChecksum);
  if (type->size() == 0) {
    while (!payload.atEnd()) {
      payload.nextView(kIoChunkSize);
    }
    presto::detail::deserializeColumns(
        nullptr,
        pool,
        type,
        result,
        header.numRows,
        resultOffset,
        serdeOptions.useLosslessTimestamp);
  } else if (!(header.flags & kCompressed)) {
    presto::detail::deserializeColumns(
        &payload,
        pool,
        type,
        result,
        header.numRows,
        resultOffset,
        serdeOptions.useLosslessTimestamp);
  } else {
    auto codec = common::compressionKindToCodec(serdeOptions.compressionKind);
    BOLT_CHECK(
        codec->type() != folly::io::CodecType::NO_COMPRESSION,
        "Compressed spill page requires a codec");
    BOLT_CHECK_LE(header.uncompressedSize, codec->maxUncompressedLength());
    // Views over spill buffers cannot outlive a refill. Copy compressed data
    // into owned storage before invoking the codec, using bounded reads.
    auto compressed = folly::IOBuf::create(header.storedSize);
    int64_t offset = 0;
    while (offset < header.storedSize) {
      const auto size = static_cast<int32_t>(
          std::min<int64_t>(kIoChunkSize, header.storedSize - offset));
      payload.readBytes(compressed->writableData() + offset, size);
      offset += size;
    }
    compressed->append(header.storedSize);
    if (header.flags & kChecksum) {
      BOLT_CHECK_EQ(
          finishChecksum(payload.crc(), header),
          header.checksum,
          "Corrupted spill page");
    }
    auto uncompressed =
        codec->uncompress(compressed.get(), header.uncompressedSize);
    BOLT_CHECK_EQ(
        uncompressed->computeChainDataLength(), header.uncompressedSize);
    std::vector<ByteRange> ranges;
    for (auto range : *uncompressed) {
      size_t pos = 0;
      while (pos < range.size()) {
        const auto size = std::min<size_t>(kIoChunkSize, range.size() - pos);
        ranges.push_back(
            {const_cast<uint8_t*>(range.data()) + pos,
             static_cast<int32_t>(size),
             0});
        pos += size;
      }
    }
    ByteInputStream columns(std::move(ranges));
    presto::detail::deserializeColumns(
        &columns,
        pool,
        type,
        result,
        header.numRows,
        resultOffset,
        serdeOptions.useLosslessTimestamp);
    BOLT_CHECK(columns.atEnd(), "Spill payload length mismatch");
  }
  BOLT_CHECK(payload.atEnd(), "Spill payload length mismatch");
  if (header.flags & kChecksum) {
    BOLT_CHECK_EQ(
        finishChecksum(payload.crc(), header),
        header.checksum,
        "Corrupted spill page");
  }
}

} // namespace bytedance::bolt::serializer
