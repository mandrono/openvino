// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/detection_output.hpp"

using namespace LayerTestsDefinitions;

namespace {

const size_t numClasses = 11;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = { {50}, {100} };
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const std::vector<size_t> numberBatch = {1, 2};

const auto commonAttributes = ::testing::Combine(
        ::testing::Values(numClasses),
        ::testing::ValuesIn(topK),
        ::testing::ValuesIn(keepTopK),
        ::testing::ValuesIn(codeType),
        ::testing::ValuesIn(clipAfterNms),
        ::testing::ValuesIn(clipBeforeNms),
        ::testing::ValuesIn(decreaseLabelId)
);

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {}, {}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {}, {}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {}, {}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {}, {}}
};

const auto params3Inputs = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams3In),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_DetectionOutput3In, DetectionOutputLayerTest, params3Inputs, DetectionOutputLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams5In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 660}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 660}}
};

const auto params5Inputs = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams5In),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_DetectionOutput5In, DetectionOutputLayerTest, params5Inputs, DetectionOutputLayerTest::getTestCaseName);

}  // namespace
