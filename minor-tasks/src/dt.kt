import java.io.BufferedReader

fun main() {
    //    val bufferedReader: BufferedReader = File("data/mf-input.txt").bufferedReader()
    val bufferedReader: BufferedReader = System.`in`.bufferedReader()
    val (m, k, h) = bufferedReader.readLine().split(' ').map(String::toInt)
    val n = bufferedReader.readLine().toInt()
    val X = ArrayList<ArrayList<Int>>()
    val y = ArrayList<Int>()
    for (i in 0 until n){
        val tmp = ArrayList<Int>()
        val line = bufferedReader.readLine().split(' ').map(String::toInt)
        tmp.addAll(line.dropLast(1))
        X.add(tmp)
        y.add(tmp.last())
    }

}