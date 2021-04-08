import kotlin.math.ln

fun main() {
    val kXkY = readLine()?.split(' ')?.toList() ?: return
    val kx = kXkY[0].toInt()
    val p = MutableList(kx) { 0.0 }
    val all = mutableMapOf<Pair<Int, Int>, Double>()
    val n = Integer.valueOf(readLine())
    var ans = 0.0
    (0 until n).forEach { _ ->
        val input = readLine()?.split(' ')?.toList() ?: return
        val x1 = input[0].toInt()
        val x2 = input[1].toInt()
        p[x1 - 1] += 1.0 / n
        all.putIfAbsent(Pair(x1 - 1, x2 - 1), 0.0)
        all[Pair(x1 - 1, x2 - 1)] = all[Pair(x1 - 1, x2 - 1)]!! + 1.0 / n
    }
    for (e in all) {
        ans -= e.value * ln(e.value / p[e.key.first])
    }
    print(ans)
}