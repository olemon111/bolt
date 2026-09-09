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
#pragma once

#include "bolt/vector/VectorStream.h"

namespace bytedance::bolt::serializer {

/// Private, versioned spill format with two fixed 64-bit page lengths.
/// This is not a Presto wire format. Files require a matching spill reader;
/// standard Presto, exchange and trace consumers retain their original serde.
class SpillVectorSerde : public VectorSerde {
 public:
  SpillVectorSerde() : VectorSerde(Kind::kSpill) {}

  /// The instance is independent of the default and named serde registries.
  static SpillVectorSerde* get();

  void estimateSerializedSize(
      VectorPtr vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch) override;

  void estimateSerializedSize(
      VectorPtr vector,
      folly::Range<const vector_size_t*> rows,
      vector_size_t** sizes,
      Scratch& scratch) override;

  std::unique_ptr<VectorSerializer> createSerializer(
      RowTypePtr type,
      int32_t numRows,
      StreamArena* streamArena,
      const Options* options = nullptr) override;

  bool supportsAppendInDeserialize() const override {
    return true;
  }

  void deserialize(
      ByteInputStream* source,
      memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      const Options* options = nullptr) override {
    deserialize(source, pool, std::move(type), result, 0, options);
  }

  void deserialize(
      ByteInputStream* source,
      memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      vector_size_t resultOffset,
      const Options* options = nullptr) override;
};

} // namespace bytedance::bolt::serializer
