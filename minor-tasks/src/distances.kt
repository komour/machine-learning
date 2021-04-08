fun main() {
    val k = Integer.valueOf(readLine())
    val n = Integer.valueOf(readLine())
    val x1 = mutableListOf<Long>()
    val x2x1 = MutableList(k) {
        mutableListOf<Long>()
    }
    for (i in 0 until n) {
        val input = readLine()?.split(' ')?.toList() ?: return
        val p = Pair(input[0].toLong(), input[1].toInt())
        x1.add(p.first)
        x2x1[p.second - 1].add(p.first)
    }
    x1.sort()
    var d1: Long = 0
    var d2: Long = 0
    for (i in x1.indices) {
        val b = n - 1
        d1 += 2 * i * x1[i]
        d1 -= 2 * x1[i] * b
        d1 += i * 2 * x1[i]
    }
    for (e in x2x1) {
        e.sort()
        for (i in e.indices) {
            val b = e.size - 1
            d2 += 2 * i * e[i]
            d2 -= 2 * e[i] * b
            d2 += i * 2 * e[i]
        }
    }
    println(d2)
    print(d1 - d2)
}