import kotlin.math.pow

fun main() {
    val k = Integer.valueOf(readLine())
    val n = Integer.valueOf(readLine())
    val xy = MutableList(k) { mutableListOf<Double>() }
    val sum = MutableList(k) { 0.0 }
    (0 until n).forEach { _ ->
        val input = readLine()?.split(' ')?.toList() ?: return
        val x = input[0].toInt()
        val y = input[1].toDouble()
        sum[x - 1] += y
        xy[x - 1].add(y)
    }
    var res = 0.0
    for (i in xy.indices) {
        var t = 0.0
        for (v in xy[i]) {
            t += (v - sum[i] / xy[i].size).pow(2)
        }
        res += t / n
    }
    println(res)
}