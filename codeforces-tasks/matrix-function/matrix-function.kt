import java.io.BufferedReader
import java.io.File
import kotlin.math.cosh
import kotlin.math.pow
import kotlin.math.sinh
import kotlin.math.tanh

fun readNumbers(bufferedReader: BufferedReader): List<Double> {
    return bufferedReader.readLine().split(' ').map(String::toDouble)
}

fun readMatrix(bufferedReader: BufferedReader, r: Int, c: Int): Array<Array<Double>> {
    val data: Array<Array<Double>> = Array(r) { Array(c) { 0.0 } }
    for (i in 0 until r) {
        val line = readNumbers(bufferedReader)
        for (j in line.indices) {
            data[i][j] = line[j]
        }
    }
    return data
}

fun printMatrix(data: Array<Array<Double>>) {
    for (row in data) {
        for (x in row) {
            print("$x ")
        }
        println()
    }
}

fun sumMatrix(first: Array<Array<Double>>, second: Array<Array<Double>>): Array<Array<Double>> {
    val res = Array(first.size) { Array(first[0].size) { 0.0 } }
    for (i in first.indices) {
        for (j in first[i].indices) {
            res[i][j] = first[i][j] + second[i][j]
        }
    }
    return res
}


fun mulMatrix(first: Array<Array<Double>>, second: Array<Array<Double>>): Array<Array<Double>> {
    val res = Array(first.size) { Array(second[0].size) { 0.0 } }
    for (i in res.indices) {
        for (j in res[0].indices) {
            for (k in second.indices) {
                res[i][j] += first[i][k] * second[k][j]
            }
        }
    }
    return res
}

fun adamMatrix(first: Array<Array<Double>>, second: Array<Array<Double>>): Array<Array<Double>> {
//    println()
//    println("first")
//    printMatrix(first)
//    println("second")
//    printMatrix(second)
//    println()
    val res = Array(first.size) { Array(first[0].size) { 1.0 } }
    for (i in res.indices) {
        for (j in res[0].indices) {
            res[i][j] = first[i][j] * second[i][j]
        }
    }
    return res
}

fun transMatrix(matrix: Array<Array<Double>>): Array<Array<Double>> {
    val res = Array(matrix[0].size) { Array(matrix.size) { 0.0 } }
    for (i in res.indices) {
        for (j in res[0].indices) {
            res[i][j] = matrix[j][i]
        }
    }
    return res
}


class VertData(val r: Int, val c: Int, val flag: Boolean = false) {
    var matrix: Array<Array<Double>> = if (flag) Array(r) { Array(c) { 1.0 } } else Array(r) { Array(c) { 0.0 } }
    var derivative: Array<Array<Double>> = if (flag) Array(r) { Array(c) { 0.0 } } else Array(r) { Array(c) { 0.0 } }
}


open class Var(open val flag: Boolean = false, open val data: VertData = if (flag) VertData(0, 0) else VertData(0, 0)) {

    open fun calculateOutput() {}
    open fun recalculateDerivative() {}
}

fun dTanh(x: Double): Double {
    return (cosh(x).pow(2) - sinh(x).pow(2)) / cosh(x).pow(2)
}

class Tnh @JvmOverloads constructor(
    private val ev: VertData,
    override val data: VertData = VertData(ev.r, ev.c)
) : Var() {
    override fun calculateOutput() {
        for (i in 0 until data.r) {
            for (j in 0 until data.c) {
                data.matrix[i][j] = tanh(ev.matrix[i][j])
            }
        }
    }

    override fun recalculateDerivative() {
        for (i in 0 until data.r) {
            for (j in 0 until data.c) {
                ev.derivative[i][j] += data.derivative[i][j] * dTanh(ev.matrix[i][j])
            }
        }
//        println("TNH")
//        printMatrix(data.derivative)
//        println()
    }
}

class Relu @JvmOverloads constructor(
    private val a: Int,
    private val ev: VertData,
    override val data: VertData = VertData(ev.r, ev.c)
) : Var() {
    override fun calculateOutput() {
        for (i in 0 until data.r) {
            for (j in 0 until data.c) {
                data.matrix[i][j] = if (ev.matrix[i][j] >= 0) ev.matrix[i][j] else ev.matrix[i][j] / a
            }
        }
    }

    override fun recalculateDerivative() {
        for (i in 0 until data.r) {
            for (j in 0 until data.c) {
                ev.derivative[i][j] += if (ev.matrix[i][j] >= 0) data.derivative[i][j] else data.derivative[i][j] / a
            }
        }
//        println("RELU")
//        printMatrix(data.derivative)
//        println()
    }
}

class Mul @JvmOverloads constructor(
    private val ev1: VertData,
    private val ev2: VertData,
    override val data: VertData = VertData(ev1.r, ev2.c)
) : Var() {
    override fun calculateOutput() {
        data.matrix = mulMatrix(ev1.matrix, ev2.matrix)
    }

    override fun recalculateDerivative() {
        ev1.derivative = sumMatrix(ev1.derivative, transMatrix(mulMatrix(ev2.matrix, transMatrix(data.derivative))))
        ev2.derivative = sumMatrix(ev2.derivative, mulMatrix(transMatrix(ev1.matrix), data.derivative))
//        println("MUL")
//        printMatrix(data.derivative)
//        println()
    }
}

class Sum @JvmOverloads constructor(
    private val vertDataList: ArrayList<VertData>,
    override val data: VertData = VertData(vertDataList[0].r, vertDataList[0].c)
) : Var(false) {
    override fun calculateOutput() {
        for (m in vertDataList) {
            data.matrix = sumMatrix(data.matrix, m.matrix)
        }
    }

    override fun recalculateDerivative() {
        for (m in vertDataList) {
            m.derivative = sumMatrix(data.derivative, m.derivative)
        }
//        println("SUM")
//        printMatrix(data.derivative)
//        println()
    }
}

class Had @JvmOverloads constructor(
    private val vertDataList: ArrayList<VertData>,
    override val data: VertData = VertData(vertDataList[0].r, vertDataList[0].c, true)
) : Var(true) {

    override fun calculateOutput() {
//        printMatrix(data.matrix)
        for (vertData in vertDataList) {
            data.matrix = adamMatrix(data.matrix, vertData.matrix)
        }
//        print("HERE ")
//        printMatrix(data.matrix)
    }

    override fun recalculateDerivative() {
        for (i in vertDataList.indices) {
            var m: Array<Array<Double>> = Array(data.r) { Array(data.c) { 1.0 } }
            for (j in vertDataList.indices) {
                if (i == j) continue
                m = adamMatrix(m, vertDataList[j].matrix)
            }
            vertDataList[i].derivative = sumMatrix(vertDataList[i].derivative, adamMatrix(m, data.derivative))
        }
//        println("HAD")
//        printMatrix(data.derivative)
//        println()
    }
}




fun main() {
//    val bufferedReader: BufferedReader = File("data/mf-input.txt").bufferedReader()
    val bufferedReader: BufferedReader = System.`in`.bufferedReader()
    val nmk = bufferedReader.readLine().split(' ').toList()
    val n = nmk[0].toInt()
    val m = nmk[1].toInt()
    val k = nmk[2].toInt()
    val vertices = mutableListOf<Var>()

    for (i in 0 until n) {
        val line = bufferedReader.readLine().split(' ').toList()
        when (line[0]) {
            "var" -> {
                val r = line[1].toInt()
                val c = line[2].toInt()
                vertices.add(Var(false, VertData(r, c)))
            }
            "tnh" -> {
                val idx = line[1].toInt()
                vertices.add(Tnh(vertices[idx - 1].data))
            }
            "rlu" -> {
                val a = line[1].toInt()
                val idx = line[2].toInt()
                vertices.add(Relu(a, vertices[idx - 1].data))
            }
            "mul" -> {
                val a = line[1].toInt()
                val b = line[2].toInt()
                vertices.add(Mul(vertices[a - 1].data, vertices[b - 1].data))
            }
            "sum" -> {
                val l = line.toMutableList()
                l.removeFirst()
                l.removeFirst()
                val matrices = arrayListOf<VertData>()
                for (j in l.indices) {
                    val idx = l[j].toInt()
                    matrices.add(vertices[idx - 1].data)
                }
                vertices.add(Sum(matrices))
            }
            "had" -> {
                val l = line.toMutableList()
                l.removeFirst()
                l.removeFirst()
                val matrices = arrayListOf<VertData>()
                for (j in l.indices) {
                    val idx = l[j].toInt()
                    matrices.add(vertices[idx - 1].data)
                }
                vertices.add(Had(matrices))
            }
        }
    }

    for (i in 0 until m) {
        vertices[i].data.matrix = readMatrix(bufferedReader, vertices[i].data.r, vertices[i].data.c)
    }
    for (i in m until n) {
        vertices[i].calculateOutput()
    }
    for (i in n - k until n) {
        vertices[i].data.derivative = readMatrix(bufferedReader, vertices[i].data.r, vertices[i].data.c)
        printMatrix(vertices[i].data.matrix)
    }
    for (i in n - 1 downTo 0) {
        vertices[i].recalculateDerivative()
    }
    for (i in 0 until m) {
        printMatrix(vertices[i].data.derivative)
    }
}