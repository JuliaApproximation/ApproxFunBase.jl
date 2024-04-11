module ApproxFunBaseStatisticsExt

import Statistics: mean
using ApproxFunBase
using ApproxFunBase: IntervalOrSegment

mean(d::IntervalOrSegment) = ApproxFunBase.mean(d)

end
