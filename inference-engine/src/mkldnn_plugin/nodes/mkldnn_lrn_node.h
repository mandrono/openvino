// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNLrnNode : public MKLDNNNode {
public:
    MKLDNNLrnNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNLrnNode() override = default;

    void getSupportedDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    bool isAcrossMaps = false;
    int size = 1;
    int k = 1;
    float alpha = 1.0f;
    float beta = 1.0f;
};

}  // namespace MKLDNNPlugin

