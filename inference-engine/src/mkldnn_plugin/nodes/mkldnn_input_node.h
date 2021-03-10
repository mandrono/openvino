// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNInputNode : public MKLDNNNode {
public:
    MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNInputNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void execute(mkldnn::stream strm) override;
    void withMeanImage() {
        isMeanImage = true;
    }

    const InferenceEngine::Blob::CPtr getConstBlob() const {
        return constBlob;
    }

private:
    InferenceEngine::Precision precision;

    InferenceEngine::Blob::Ptr constBlob;
    bool isMeanImage = false;
};

}  // namespace MKLDNNPlugin

