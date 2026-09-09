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

namespace bytedance::bolt::serializer::presto::detail {

/// Internal access to the column payload, without a Presto page header.
/// The standard Presto serializer and the internal spill format share column
/// encoding, but own their page framing independently.
class ColumnSerializer : public VectorSerializer {
 public:
  virtual int32_t numRows() const = 0;
  virtual size_t columnsSize() const = 0;
  virtual void flushColumns(OutputStream* out) = 0;
};

std::unique_ptr<ColumnSerializer> createColumnSerializer(
    const RowTypePtr& type,
    int32_t numRows,
    StreamArena* arena,
    bool useLosslessTimestamp);

/// Restores column data, including result reuse, append and nested row nulls.
/// An empty projection only prepares the result; the caller consumes the page.
void deserializeColumns(
    ByteInputStream* source,
    memory::MemoryPool* pool,
    const RowTypePtr& type,
    RowVectorPtr* result,
    int32_t numRows,
    vector_size_t resultOffset,
    bool useLosslessTimestamp);

} // namespace bytedance::bolt::serializer::presto::detail
