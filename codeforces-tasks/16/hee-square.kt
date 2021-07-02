import kotlin.math.pow

fun main() {
    val k1k2 = readLine()?.split(' ')?.toList() ?: return
    val k1 = k1k2[0].toInt()
    val k2 = k1k2[1].toInt()
    val eX = MutableList(k1) { 0.0 }
    val eY = MutableList(k2) { 0.0 }
    val all = mutableMapOf<Pair<Int, Int>, Double>()
    val n = Integer.valueOf(readLine())
    var hee2 = n.toDouble()
    (0 until n).forEach { _ ->
        val input = readLine()?.split(' ')?.toList() ?: return
        val x1 = input[0].toInt()
        val x2 = input[1].toInt()
        eX[x1 - 1] += 1.0 / n
        eY[x2 - 1] += 1.0 / n
        all.putIfAbsent(Pair(x1 - 1, x2 - 1), 0.0)
        all[Pair(x1 - 1, x2 - 1)] = all[Pair(x1 - 1, x2 - 1)]!! + 1.0
    }
    for (p in all) {
        val ej = n * eX[p.key.first] * eY[p.key.second]
        hee2 -= ej - (p.value - ej).pow(2) / ej
    }
    print(hee2)
}