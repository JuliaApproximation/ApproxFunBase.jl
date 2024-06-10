module ApproxFunBaseTest

include("testutils.jl")
using .TestUtils

# These routines are for the unit tests

export testspace, testfunctional, testraggedbelowoperator, testbandedblockbandedoperator,
    testbandedoperator, testtransforms, testcalculus, testmultiplication, testinfoperator,
    testblockbandedoperator, testbandedbelowoperator

end

using .ApproxFunBaseTest
