// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/normalize_l2.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

using MatMulLayerTestParams = std::tuple<SizeVector,
                                         SizeVector,
                                         Precision,
                                         helpers::InputLayerType,
                                         bool,
                                         bool>;

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParams,
                                              std::string,
                                              fusingSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulLayerCPUTestParamSet> obj) {
        MatMulLayerTestParams basicParamsSet;
        fusingSpecificParams fusingParams;
        std::string layerType;
        std::tie(basicParamsSet, layerType, fusingParams) = obj.param;

        SizeVector isA, isB;
        bool transpA, transpB;
        Precision prec;
        helpers::InputLayerType typeB;
        std::tie(isA, isB, prec, typeB, transpA, transpB) = basicParamsSet;

        std::ostringstream result;
        result << layerType << "_";
        result << "IS_A=" << CommonTestUtils::vec2str(isA) << "_";
        result << "IS_B=" << CommonTestUtils::vec2str(isB) << "_";
        result << "Transp_A=" << transpA << "_";
        result << "Transp_B=" << transpB << "_";
        result << "Prec=" << prec << "_";
        result << "typeB=" << typeB;

        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    std::string layerType;

    void SetUp() override {
        MatMulLayerTestParams basicParamsSet;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, layerType, fusingParams) = this->GetParam();
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        SizeVector isA, isB;
        bool transpA, transpB;
        Precision prec;
        helpers::InputLayerType typeB;
        std::tie(isA, isB, prec, typeB, transpA, transpB) = basicParamsSet;

        auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prec);
        auto params = builder::makeParams(ngPrec, {isA});
        auto matrixB = builder::makeInputLayer(ngPrec, typeB, isB);
        if (typeB == helpers::InputLayerType::PARAMETER) {
            params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
        }
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));
        auto matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);

        function = makeNgraphFunction(ngPrec, params, matMul, layerType);
    }
};

TEST_P(MatMulLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, layerType);
}

namespace {

/* ============= Common params ============= */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec
};

const std::vector<std::pair<SizeVector, SizeVector>> inputShapes = {
    {{}, {}}
};

const std::vector<bool> transpose = {
    true, false
};

std::vector<helpers::InputLayerType> typeB = {
        helpers::InputLayerType::CONSTANT,
        helpers::InputLayerType::PARAMETER,
};

const auto matMuleParams = ::testing::Combine(::testing::ValuesIn(axes_4D),
                                                   ::testing::Values(epsilon),
                                                   ::testing::Values(epsMode),
                                                   ::testing::ValuesIn(inputShape_4D),
                                                   ::testing::ValuesIn(netPrecisions),
                                                   ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto testParams = ::testing::Combine(normalizeParams_4D,
                                              ::testing::ValuesIn(getCPUSpecificParams()),
                                              ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_CASE_P(smoke_Check, NormalizeL2LayerCPUTest, testParams_4D, NormalizeL2LayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
