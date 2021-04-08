//import java.io.BufferedReader
//import java.io.File
//import kotlin.collections.ArrayList
//import kotlin.math.pow
//import kotlin.math.tanh
//
//fun readNumbers(bufferedReader: BufferedReader): List<Double> {
//    return bufferedReader.readLine().split(' ').map(String::toDouble)
//}
//
//fun readMatrix(bufferedReader: BufferedReader, r: Int, c: Int): Array<Array<Double>> {
//    val data: Array<Array<Double>> = Array(r) { Array(c) { 0.0 } }
//    for (i in 0 until r) {
//        val line = readNumbers(bufferedReader)
//        for (j in line.indices) {
//            data[i][j] = line[j]
//        }
//    }
//    return data
//}
//
//fun printMatrix(data: Array<Array<Double>>) {
//    for (row in data) {
//        for (x in row) {
//            print("$x ")
//        }
//        println()
//    }
//}
//
//fun sumMatrix(first: Array<Array<Double>>, second: Array<Array<Double>>) {
//    val res = Array(first.size) { Array(first[0].size) { 0.0 } }
//    for (i in first.indices) {
//        for (j in first[i].indices) {
//            res[i][j] = first[i][j] + second[i][j]
//        }
//    }
//}
//
//
//class Matrix(val r: Int, val c: Int) {
//    var matrix: Array<Array<Double>> = Array(r) { Array(c) { 0.0 } }
//    var derivative: Array<Array<Double>> = Array(r) { Array(c) { 0.0 } }
//}
//
//
//open class Var(internal open val data: Matrix = Matrix(0, 0)) {
//    open fun calculateOutput() {}
//    open fun recalculateDerivative() {}
//}
//
//
//class Tnh @JvmOverloads constructor(
//    private val ev: Matrix,
//    override val data: Matrix = Matrix(ev.r, ev.c)
//) : Var() {
//    override fun calculateOutput() {
//        for (i in 0 until data.r) {
//            for (j in 0 until data.c) {
//                data.matrix[i][j] = tanh(ev.matrix[i][j])
//            }
//        }
//    }
//
//    override fun recalculateDerivative() {
//        for (i in 0 until data.r) {
//            for (j in 0 until data.c) {
//                ev.derivative[i][j] += data.derivative[i][j] * (1 - data.matrix[i][j].pow(2))
//            }
//        }
//    }
//}
//
//class Rlu @JvmOverloads constructor(
//    private val a: Int,
//    private val ev: Matrix,
//    override val data: Matrix = Matrix(ev.r, ev.c)
//) : Var() {
//    override fun calculateOutput() {
//        for (i in 0 until data.r) {
//            for (j in 0 until data.c) {
//                data.matrix[i][j] = if (ev.matrix[i][j] >= 0) ev.matrix[i][j] else ev.matrix[i][j] / a
//            }
//        }
//    }
//
//    override fun recalculateDerivative() {
//        for (i in 0 until data.r) {
//            for (j in 0 until data.c) {
//                ev.derivative[i][j] += if (ev.matrix[i][j] >= 0) data.derivative[i][j] else data.derivative[i][j] / a
//            }
//        }
//    }
//}
//
//class Mul @JvmOverloads constructor(
//    private val ev1: Matrix,
//    private val ev2: Matrix,
//    override val data: Matrix = Matrix(ev1.r, ev2.c)
//) : Var() {
//    override fun calculateOutput() {
//        for (i in 0 until data.r) {
//            for (j in 0 until data.c) {
//                for (k in 0 until ev2.r) {
//                    data.matrix[i][j] += ev1.matrix[i][k] * ev2.matrix[k][j]
//                }
//            }
//        }
//    }
//
//    override fun recalculateDerivative() {  // TODO
//        for (i in 0 until ev1.r) {
//            for (j in 0 until ev1.c) {
//                for (k in 0 until ev2.c) {
//                    ev1.derivative[i][j] += data.derivative[i][k] * ev2.matrix[j][k]
//                }
//            }
//        }
//
//        for (i in 0 until ev1.c) {
//            for (j in 0 until ev2.c) {
//                for (k in 0 until ev1.r) {
//                    ev2.derivative[i][j] += ev1.matrix[k][i] * data.derivative[k][j]
//                }
//            }
//        }
//    }
//}
//
//class Sum(
//    private val matrixList: ArrayList<Matrix>,
//    override val data: Matrix = Matrix(matrixList[0].r, matrixList[0].c)
//) : Var() {
//    override fun calculateOutput() {
//        for (m in matrixList) {
//            for (i in 0 until data.r) {
//                for (j in 0 until data.c) {
//                    data.matrix[i][j] += m.matrix[i][j]
//                }
//            }
//        }
//    }
//
//    override fun recalculateDerivative() {
//        for (m in matrixList) {
//            for (i in 0 until data.r) {
//                for (j in 0 until data.c) {
//                    m.derivative[i][j] += data.derivative[i][j]
//                }
//            }
//        }
//    }
//}
//
//class Had(
//    private val matrixList: ArrayList<Matrix>,
//    override val data: Matrix = Matrix(matrixList[0].r, matrixList[0].c)
//) : Var() {
//
//    override fun calculateOutput() {
//        for (matrix in matrixList) {
//            for (i in 0 until data.r) {
//                for (j in 0 until data.c) {
//                    data.matrix[i][j] *= matrix.matrix[i][j]
//                }
//            }
//        }
//    }
//
//    override fun recalculateDerivative() {  // TODO
//        for (matrix in matrixList) {
//            for (i in 0 until data.r) {
//                for (j in 0 until data.c) {
//                    var k = 1.0
//                    for (m in matrixList) {
//                        if (m == matrix) continue
//                        k *= m.matrix[i][j]
//                    }
//                    matrix.derivative[i][j] += k * data.derivative[i][j]
//                }
//            }
//        }
//    }
//}
//
//
//fun main() {
////    val bufferedReader: BufferedReader = File("data/mf-input.txt").bufferedReader()
//    val bufferedReader: BufferedReader = System.`in`.bufferedReader()
//    val nmk = bufferedReader.readLine().split(' ').toList()
//    val n = nmk[0].toInt()
//    val m = nmk[1].toInt()
//    val k = nmk[2].toInt()
//    val vertices = mutableListOf<Var>()
//
//    for (i in 0 until n) {
//        val line = bufferedReader.readLine().split(' ').toList()
//        when (line[0]) {
//            "var" -> {
//                val r = line[1].toInt()
//                val c = line[2].toInt()
//                vertices.add(Var(Matrix(r, c)))
//            }
//            "tnh" -> {
//                val idx = line[1].toInt()
//                vertices.add(Tnh(vertices[idx - 1].data))
//            }
//            "rlu" -> {
//                val a = line[1].toInt()
//                val idx = line[2].toInt()
//                vertices.add(Rlu(a, vertices[idx - 1].data))
//            }
//            "mul" -> {
//                val a = line[1].toInt()
//                val b = line[2].toInt()
//                vertices.add(Mul(vertices[a - 1].data, vertices[b - 1].data))
//            }
//            "sum" -> {
//                val l = line.toMutableList()
//                l.removeFirst()
//                l.removeFirst()
//                val matrices = arrayListOf<Matrix>()
//                for (j in l.indices) {
//                    val idx = l[j].toInt()
//                    matrices.add(vertices[idx - 1].data)
//                }
//                vertices.add(Sum(matrices))
//            }
//            "had" -> {
//                val l = line.toMutableList()
//                l.removeFirst()
//                l.removeFirst()
//                val matrices = arrayListOf<Matrix>()
//                for (j in l.indices) {
//                    val idx = l[j].toInt()
//                    matrices.add(vertices[idx - 1].data)
//                }
//                vertices.add(Had(matrices))
//            }
//        }
//    }
//    for (i in 0 until m) {
//        vertices[i].data.matrix = readMatrix(bufferedReader, vertices[i].data.r, vertices[i].data.c)
//    }
//    for (i in m until n) {
//        vertices[i].calculateOutput()
//    }
//    for (i in n - k until n) {
//        vertices[i].data.derivative = readMatrix(bufferedReader, vertices[i].data.r, vertices[i].data.c)
//        printMatrix(vertices[i].data.matrix)
//    }
//    for (i in n - 1 downTo 0) {
//        vertices[i].recalculateDerivative()
//    }
//    for (i in 0 until m) {
//        printMatrix(vertices[i].data.derivative)
//    }
//}