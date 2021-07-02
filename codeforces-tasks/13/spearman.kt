import kotlin.math.abs
import kotlin.math.pow

fun main() {
    val eps = 1e-7
    val n = Integer.valueOf(readLine())
    val featuresListX1 = mutableListOf<Pair<Double, Int>>()
    val featuresListX2 = mutableListOf<Pair<Double, Int>>()
    for (i in 0 until n) {
        val input = readLine()?.split(' ')?.toList() ?: return
        val p = Pair(input[0].toDouble(), input[1].toDouble())
        featuresListX1.add(Pair(p.first, i))
        featuresListX2.add(Pair(p.second, i))
    }
    featuresListX1.sortBy { p -> p.first }
    featuresListX2.sortBy { p -> p.first }

    val rankX1 = MutableList(n) { 0.0 }
    val rankX2 = MutableList(n) { 0.0 }

    var num = 0.0

    for (i in 1 until n) {
        rankX1[featuresListX1[i].second] = i.toDouble()
        rankX2[featuresListX2[i].second] = i.toDouble()
    }
    for (i in 0 until n) {
        num += (rankX1[i] - rankX2[i]).pow(2)
    }
    num *= 6
    val den = n * (n - 1.0) * (n + 1)

    when {
        abs(num) < eps -> print(0)
        den == 0.0 -> print(0)
        else -> print(1 - num / den)
    }
}