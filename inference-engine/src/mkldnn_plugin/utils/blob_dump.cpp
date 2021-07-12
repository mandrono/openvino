// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_dump.h"
#include "blob_factory.hpp"
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"
#include <nodes/common/cpu_memcpy.h>

#include "common/memory_desc_wrapper.hpp"

#include <fstream>
#include <cpu_memory_desc_utils.h>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

// IEB file format routine
static unsigned char IEB_MAGIC[4] = {'I', 'E', 'B', '0'};
static unsigned char NO_SCALES = 0xFF;

struct IEB_HEADER {
    unsigned char magic[4];
    unsigned char ver[2];

    unsigned char precision;  // 0-8
    unsigned char ndims;
    unsigned int  dims[7];  // max is 7-D blob

    unsigned char scaling_axis;  // FF - no scaling
    unsigned char reserved[3];

    unsigned long data_offset;
    unsigned long data_size;
    unsigned long scaling_data_offset;
    unsigned long scaling_data_size;
};

static IEB_HEADER prepare_header(const MKLDNNMemoryDesc& desc) {
    IEB_HEADER header = {};

    header.magic[0] = IEB_MAGIC[0];
    header.magic[1] = IEB_MAGIC[1];
    header.magic[2] = IEB_MAGIC[2];
    header.magic[3] = IEB_MAGIC[3];

    // IEB file format version 0.1
    header.ver[0] = 0;
    header.ver[1] = 1;

    header.precision = desc.getPrecision();

    if (desc.getShape().getRank() > 7)
        IE_THROW() << "Dumper support max 7D blobs";

    header.ndims = desc.getShape().getRank();
    const auto &dims = desc.getShape().getStaticDims();
    for (int i = 0; i < header.ndims; i++)
        header.dims[i] = dims[i];

    header.scaling_axis = NO_SCALES;

    return header;
}

static MKLDNNMemoryDesc parse_header(IEB_HEADER &header) {
    if (header.magic[0] != IEB_MAGIC[0] ||
        header.magic[1] != IEB_MAGIC[1] ||
        header.magic[2] != IEB_MAGIC[2] ||
        header.magic[3] != IEB_MAGIC[3])
        IE_THROW() << "Dumper cannot parse file. Wrong format.";

    if (header.ver[0] != 0 ||
        header.ver[1] != 1)
        IE_THROW() << "Dumper cannot parse file. Unsupported IEB format version.";

    const auto prc = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision(static_cast<Precision::ePrecision>(header.precision)));
    SizeVector dims(header.ndims);
    for (int i = 0; i < header.ndims; i++)
        dims[i] = header.dims[i];

    return MKLDNNMemoryDesc{MKLDNNDims(dims), prc, MKLDNNMemory::GetPlainFormatByRank(dims.size()) };
}

static void prepare_plain_data(const MKLDNNMemoryDesc &mdesc, const void *ptr, std::vector<uint8_t> &data) {
    const auto size = mdesc.getDims().size() * mdesc.GetElementSize();
    data.resize(size);
    mkldnn::memory::desc desc = mdesc;
    mkldnn::impl::memory_desc_wrapper mem_wrp(desc.data);


    // check if it already plain
    if (mdesc.checkGeneralLayout(GeneralLayout::ncsp)) {
        cpu_memcpy(data.data(), reinterpret_cast<const uint8_t*>(ptr) + mem_wrp.offset0() * mem_wrp.data_type_size(), size);
        return;
    }

    // Copy to plain
    size_t data_size = mdesc.getDims().size();

    switch (mdesc.getPrecision()) {
        case Precision::FP32:
        case Precision::I32: {
            auto *pln_blob_ptr = reinterpret_cast<int32_t *>(data.data());
            auto *blob_ptr = reinterpret_cast<const int32_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        case Precision::BF16: {
            auto *pln_blob_ptr = reinterpret_cast<int16_t *>(data.data());
            auto *blob_ptr = reinterpret_cast<const int16_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        case Precision::I8:
        case Precision::U8: {
            auto *pln_blob_ptr = reinterpret_cast<int8_t*>(data.data());
            auto *blob_ptr = reinterpret_cast<const int8_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }
}

void BlobDumper::dump(std::ostream &stream) const {
    if (ptr == nullptr)
        IE_THROW() << "Dumper cannot dump. Memory is not allocated.";

    IEB_HEADER header = prepare_header(this->desc);
    std::vector<uint8_t> data;
    prepare_plain_data(desc, ptr, data);

    header.data_offset = sizeof(header);
    header.data_size = data.size();
    header.scaling_data_offset = 0;
    header.scaling_data_size = 0;

    if (_scales) {
        header.scaling_axis = 1;
        header.scaling_data_offset = header.data_offset + header.data_size;
        header.scaling_data_size = _scales->byteSize();
    }

    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<char*>(data.data()), data.size());

    if (_scales) {
        stream.write(_scales->buffer().as<char*>(), _scales->byteSize());
    }
}

void BlobDumper::dumpAsTxt(std::ostream &stream) const {
    if (ptr == nullptr)
        IE_THROW() << "Dumper cannot dump. Memory is not allocated.";

    const auto dims = desc.getShape().getStaticDims();

    // Header like "U8 4D shape: 2 3 224 224 ()
    stream << desc.getPrecision().name() << " "
           << dims.size() << "D "
           << "shape: ";
    for (size_t d : dims) stream << d << " ";
    stream << "(" << desc.getShape().getElementsCount() << ")" <<
    " by address 0x" << std::hex << reinterpret_cast<const long long *>(ptr) << std::dec <<std::endl;

    mkldnn::memory::desc mkldnnDesc = this->desc;
    mkldnn::impl::memory_desc_wrapper mem_wrp(mkldnnDesc.data);

    size_t data_size = desc.getShape().getElementsCount();
    switch (desc.getPrecision()) {
        case InferenceEngine::Precision::FP32 : {
            auto *blob_ptr = reinterpret_cast<const float*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[mem_wrp.off_l(i)] << std::endl;
            break;
        }
        case InferenceEngine::Precision::BF16:
        {
            auto *blob_ptr = reinterpret_cast<const int16_t*>(ptr);
            for (size_t i = 0; i < data_size; i++) {
                int i16n = blob_ptr[mem_wrp.off_l(i)];
                i16n = i16n << 16;
                float fn = *(reinterpret_cast<const float *>(&i16n));
                stream << fn << std::endl;
            }
            break;
        }
        case InferenceEngine::Precision::I32: {
            auto *blob_ptr = reinterpret_cast<const int32_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[mem_wrp.off_l(i)] << std::endl;
            break;
        }
        case InferenceEngine::Precision::I8: {
            auto *blob_ptr = reinterpret_cast<const int8_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[mem_wrp.off_l(i)]) << std::endl;
            break;
        }
        case InferenceEngine::Precision::U8: {
            auto *blob_ptr = reinterpret_cast<const uint8_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[mem_wrp.off_l(i)]) << std::endl;
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }
}

BlobDumper BlobDumper::read(std::istream &stream) {
    IEB_HEADER header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));

    const auto desc = parse_header(header);

    stream.seekg(header.data_offset, stream.beg);
    std::vector<uint8_t> data(header.data_size);
    stream.read(reinterpret_cast<char *>(data.data()), header.data_size);

    BlobDumper res(desc, data.data());

    // Parse scales fields.
    if (header.scaling_axis != NO_SCALES) {
        if (header.scaling_axis != 1)
            IE_THROW() << "Dumper support scaling only for channel dims.";

        size_t scl_size = header.scaling_data_size / sizeof(float);
        auto scl = make_blob_with_precision({Precision::FP32, {scl_size}, C});
        scl->allocate();

        stream.seekg(header.scaling_data_offset, stream.beg);
        stream.read(scl->buffer().as<char*>(), header.scaling_data_size);

        res._scales = scl;
    }
    return res;
}

BlobDumper BlobDumper::read(const std::string &file_path) {
    std::ifstream file;
    file.open(file_path);
    if (!file.is_open())
        IE_THROW() << "Dumper cannot open file " << file_path;

    auto res = read(file);
    file.close();
    return res;
}

void BlobDumper::dump(const std::string &dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dump(dump_file);
    dump_file.close();
}

void BlobDumper::dumpAsTxt(const std::string& dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dumpAsTxt(dump_file);
    dump_file.close();
}

template <typename data_t>
static void plain_copy(const MKLDNNMemoryDesc &mdesc, const void *ptr, const Blob::Ptr &scls, Blob::Ptr &to) {
    auto dims = mdesc.getShape().getStaticDims();

    size_t data_size = mdesc.getDims().size() * mdesc.GetElementSize();
    size_t outer_size = dims[0];
    size_t c_size = dims.size() > 1 ? dims[1] : 1;
    size_t inner_size = dims.size() == 4 ? dims[2]*dims[3] :
                        dims.size() == 3 ? dims[2] : 1;

    auto to_data  = to->buffer().as<float*>();
    mkldnn::memory::desc desc = mdesc;
    mkldnn::impl::memory_desc_wrapper mem_wrp(desc.data);
    auto from_data = reinterpret_cast<const data_t*>(ptr) + mem_wrp.offset0();

    if (scls) {
        auto scls_data = scls->buffer().as<float*>();

        for (size_t o=0; o < outer_size; o++)
        for (size_t c=0; c < c_size; c++)
        for (size_t i=0; i < inner_size; i++)
            *to_data++ = static_cast<float>(*from_data++) * scls_data[c];
    } else {
        for (size_t i=0; i < data_size; i++)
            *to_data++ = static_cast<float>(*from_data++);
    }
}

Blob::Ptr BlobDumper::getRealValue() {
    auto res = make_plain_blob(Precision::FP32, desc.getShape().getStaticDims());
    res->allocate();

    switch (desc.getPrecision()) {
        case Precision::U8: plain_copy<uint8_t>(desc, ptr, _scales, res); break;
        case Precision::FP32: plain_copy<float>(desc, ptr, _scales, res); break;
        case Precision::I8: plain_copy<int8_t >(desc, ptr, _scales, res); break;
        default: IE_THROW() << "Unsupported precesion for getRealValue method.";
    }

    return res;
}

BlobDumper& BlobDumper::withScales(InferenceEngine::Blob::Ptr scales) {
    if (desc.getShape().getRank() < 2  ||
        scales->getTensorDesc().getDims().size() != 1 ||
        scales->getTensorDesc().getDims()[0] != desc.getShape().getStaticDims()[1] ||
        scales->getTensorDesc().getPrecision() != Precision::FP32)
        IE_THROW() << "Dumper cannot use passed scales. Blob has incompatible shape.";

    _scales = scales;
    return *this;
}

BlobDumper& BlobDumper::withoutScales() {
    _scales.reset();
    return *this;
}

const InferenceEngine::Blob::Ptr& BlobDumper::getScales() const {
    return _scales;
}

}  // namespace MKLDNNPlugin
