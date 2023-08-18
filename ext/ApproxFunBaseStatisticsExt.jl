module ApproxFunBaseStatisticsExt

import ApproxFunBase
using ApproxFunBase: IntervalOrSegment
import Statistics: mean

mean(d::IntervalOrSegment) = ApproxFunBase.mean(d)

end
