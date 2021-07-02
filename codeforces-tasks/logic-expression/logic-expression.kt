import kotlin.math.pow

fun main() {
    val m = readInt()
    val n = 2.pow(m)
    val network = mutableListOf<MutableList<Double>>()
    for (i in 0 until n) {
        val f = readInt()
        if (f == 0) continue
        val cond = mutableListOf<Double>()
        var v = 0.5
        (0 until m).forEach { j ->
            val z = 1 shl j
            cond.add(
                if (i and z != 0) {
                    1.0
                } else {
                    -1337.42
                }
            )
            v -= if (cond.last() != 1.0) {
                0
            } else {
                1
            }
        }
        cond.add(v)
        network.add(cond)
    }
    if (network.isEmpty()) {
        println("1\n1")
        (0 until m).forEach { _ -> print("0 ") }
    } else {
        println(2)
        val s = network.size.toString()
        println("$s 1")
        network.print()
        (0 until network.size).forEach { _ -> print("1.0 ") }
    }
    print(-0.5)
}

private fun MutableList<MutableList<Double>>.print() {
    for (list in this) {
        for (e in list) {
            print("$e ")
        }
        println()
    }
}

private fun readInt(): Int {
    return Integer.valueOf(readLine())
}

private fun Int.pow(m: Int): Int {
    val x = m.toDouble()
    val y = toDouble()
    return y.pow(x).toInt()
}
