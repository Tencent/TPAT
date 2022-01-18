##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
# Not Supported int8、float16 and float64 type.
unsupported_ops = {
    "ConvInteger",
    "CumSum",
    "DequantizeLinear",
    "Det",
    "LpNormalization",
    "MatMulInteger",
    "MeanVarianceNormalization",
    "Multinomial",
    "NonMaxSuppression",
    "NonZero",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SplitToSequence",
    "StringNormalizer",
    "TfIdfVectorizer",
    "TopK",
    "Trilu",
    "Unique",
}

no_needed_plugin_ops = {
    "Cast",
    "Constant",
    "ConstantOfShape",
    "Dropout",
    "GridSample",
    "If",
    "Loop",
    "Shape",
    "Squeeze",
    "Unsqueeze",
    "RandomUniform",
    "RandomNormalLike",
    "Scan",
}
