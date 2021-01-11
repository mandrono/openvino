// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//#include <ie_common.h>
#include <mkldnn_node.h>
//#include <string>

namespace MKLDNNPlugin {

class MKLDNNReferenceNode : public MKLDNNNode {
public:
<<<<<<< HEAD
    MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache, const std::string& errorMessage);
=======
    MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
>>>>>>> [CPU] Plug-in migration on ngraph initial commit
    ~MKLDNNReferenceNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
<<<<<<< HEAD
    const std::string additionalErrorMessage;
=======
>>>>>>> [CPU] Plug-in migration on ngraph initial commit
};

}  // namespace MKLDNNPlugin

