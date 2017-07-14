package com.xj.ml.vectormatrix

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/7
  */
object VectorsCompanionObjectDemo {
  def main(args: Array[String]): Unit = {

    // 创建密集向量类
    val dense: linalg.Vector = Vectors.dense(1.0, 2.0, 3.0)
    println(dense)          // [1.0,2.0,3.0]

    // 创建稀疏向量类(参数含义： 数据个数，下标，value)
    val sparse: linalg.Vector = Vectors.sparse(4, Array(0, 1, 3, 4), Array(1.0, 2.0, 3.0, 4.0))
    println(sparse)         // (4,[0,1,3,4],[1.0,2.0,3.0,4.0])

    // 创建稀疏向量类(参数含义: 向量大小，向量元素Seq((下标， value),()...))
    val sparse1: linalg.Vector = Vectors.sparse(5, Seq((1, 1.0), (2, 3.0), (4, 5.0)))
    println(sparse1)        // (5,[1,2,4],[1.0,3.0,5.0])

    // 创建0向量
    val zeros: linalg.Vector = Vectors.zeros(3)
    println(zeros)          // [0.0,0.0,0.0]

    // 求向量的p范数
    val norm: Double = Vectors.norm(dense, 2.0)
    println(norm)           // 3.7416573867739413

    val dense1 = Vectors.dense(1.0, 2.0, 3.0)
    val dense2 = Vectors.dense(1.0, 1.0, 1.0)
    // 求向量之间的平方距离
    val sqdist: Double = Vectors.sqdist(dense1, dense2)
    println(sqdist)         // 5.0

    // 求向量之间的平方距离（向量为稀疏矩阵和密集矩阵）
    val sv: linalg.Vector = Vectors.sparse(3,Array(0, 2), Array(1.0, 1.0))
    val dv: linalg.Vector = Vectors.dense(1.0, 1.0, 1.0)
    val sqdist1 = Vectors.sqdist(sv, dv)
    println(sqdist1)        // 1.0

    // 向量相等
    println(Vectors.dense(1.0, 2.0, 3.0).equals(Vectors.dense(1.0, 2.0, 3.0)))  // true

  }
}
