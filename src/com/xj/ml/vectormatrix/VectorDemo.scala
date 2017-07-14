package com.xj.ml.vectormatrix

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/7
  */
object VectorDemo {
  def main(args: Array[String]): Unit = {

    val vector: linalg.Vector = Vectors.dense(1.2, 1.1, 1.0)

    // 向量大小
    val size: Int = vector.size
    println(size)   // 3

    // 向量转成数组
    val toArray: Array[Double] = vector.toArray
    for(i <- 0 until(toArray.length)) println(toArray(i))  // 1.2, 1.1, 1.0

    // 返回一个向量的Hash值
    val code: Int = vector.hashCode()
    println(code)  // -1616093566

    // 向量复制
    val copy: linalg.Vector = vector.copy
    println(copy)  // [1.2,1.1,1.0]

    // 对每个元素执行函数f
    // TODO
    //vector.foreachActive()

    // 活动项数
    val actives: Int = vector.numActives
    println(actives)  // 3

    // 非零元素的数目
    val nonzeros: Int = vector.numNonzeros
    println(nonzeros) // 3

    // 转成稀疏向量
    val sparse: _root_.org.apache.spark.mllib.linalg.Vector = vector.toSparse
    println(sparse)  // (3,[0,1,2],[1.2,1.1,1.0])

    // 转成密集向量
    val dense: _root_.org.apache.spark.mllib.linalg.Vector = vector.toDense
    println(dense)   // [1.2,1.1,1.0]

    // 向量压缩（自动转化成密集向量或者稀疏向量）
    val compressed: linalg.Vector = vector.compressed
    println(compressed)  // [1.2,1.1,1.0]

  }
}





