import kotlin.math.abs
import kotlin.math.sqrt

fun main() {
    val eps = 1e-7
    val n = Integer.valueOf(readLine())
    val featuresList = mutableListOf<Pair<Double, Double>>()
    var sumX1 = 0.0
    var sumX2 = 0.0
    for (i in 1..n) {
        val input = readLine()?.split(' ')?.toList() ?: return
        val p = Pair(input[0].toDouble(), input[1].toDouble())
        featuresList.add(p)
        sumX1 += p.first
        sumX2 += p.second
    }

    val avgX1: Double = sumX1 / featuresList.size
    val avgX2: Double = sumX2 / featuresList.size

    var sum1 = 0.0
    var sum2 = 0.0
    var sum3 = 0.0
    for (p in featuresList) {
        sum1 += (p.first - avgX1) * (p.second - avgX2)
        sum2 += (p.first - avgX1) * (p.first - avgX1)
        sum3 += (p.second - avgX2) * (p.second - avgX2)
    }
    val den = sqrt(sum2 * sum3)
    when {
        abs(sum1) < eps -> {
            print(0)
        }
        den == 0.0 -> {
            print(0)
        }
        else -> {
            print(sum1 / den)
        }
    }
}