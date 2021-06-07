// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace HeteroTests {

using QueryNetworkTestParameters = std::tuple<
    std::string,                                                        // devices
    std::pair<std::set<std::string>, std::shared_ptr<ngraph::Function>> // expected nodes/graph
>;

struct QueryNetworkTest : public testing::WithParamInterface<QueryNetworkTestParameters>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
    enum {Plugin, Function};
    ~QueryNetworkTest() override = default;
    void SetUp() override;
    static std::string getTestCaseName(const ::testing::TestParamInfo<QueryNetworkTestParameters>& obj);
    std::string targetDevice;
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNetwork;
    std::set<std::string> expectedLayers;
    static std::pair<std::set<std::string>, std::shared_ptr<ngraph::Function>> generateParams(std::shared_ptr<ngraph::Function> graph);
};

}  //  namespace HeteroTests
