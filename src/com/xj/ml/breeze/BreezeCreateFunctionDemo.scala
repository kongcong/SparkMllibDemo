package com.xj.ml.breeze

import breeze.linalg.{DenseMatrix, DenseVector, Transpose, diag}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object BreezeCreateFunctionDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("BreezeCreateFunctionDemo")
    val sc = new SparkContext(conf)

    // 生成一个2行3列的矩阵
    val matrix1: DenseMatrix[Double] = DenseMatrix.zeros[Double](2, 3)
    println(matrix1)
    /**
      * 打印输出
      * 0.0  0.0  0.0
      * 0.0  0.0  0.0
      */

    // 生成一个长度为3的全0向量
    val vector1: DenseVector[Double] = DenseVector.zeros[Double](3)
    println(vector1)  // DenseVector(0.0, 0.0, 0.0)

    // 生成一个长度为3的全1向量
    val vector2: DenseVector[Double] = DenseVector.ones[Double](3)
    println(vector2)  // DenseVector(1.0, 1.0, 1.0)

    // 按指定数值填充向量
    val vector3: DenseVector[Double] = DenseVector.fill(3){5.0}
    println(vector3)  // DenseVector(5.0, 5.0, 5.0)

    // 生成随机向量 参数：start stop step
    val vector4: DenseVector[Int] = DenseVector.range(1, 10, 2)
    println(vector4)  // DenseVector(1, 3, 5, 7, 9)

    // 生成三行三列的单位矩阵
    val matrix2: DenseMatrix[Double] = DenseMatrix.eye[Double](3)
    println(matrix2)
    /**
      * 打印输出
      * 1.0  0.0  0.0
      * 0.0  1.0  0.0
      * 0.0  0.0  1.0
      */

    // 生成对角矩阵
    val matrix3: DenseMatrix[Double] = diag(DenseVector(1.0, 2.0, 3.0))
    println(matrix3)
    /**
      * 1.0  0.0  0.0
      * 0.0  2.0  0.0
      * 0.0  0.0  3.0
      */

    // 按照行创建矩阵
    val matrix4: DenseMatrix[Double] = DenseMatrix((1.0, 2.0),(3.0, 4.0))
    println(matrix4)
    /**
      * 1.0  2.0
      * 3.0  4.0
      */

    // 按照行创建向量
    val vector5: DenseVector[Int] = DenseVector(1, 2, 3, 4)
    println(vector5)  // DenseVector(1, 2, 3, 4)

    // 向量转置
    val vector6: Transpose[DenseVector[Int]] = DenseVector(1, 2, 3, 4).t
    println(vector6)  // Transpose(DenseVector(1, 2, 3, 4))

    // 从函数创建向量
    val vector7: DenseVector[Int] = DenseVector.tabulate(3){ i => 2 * i}
    println(vector7)  // DenseVector(0, 2, 4)

    // 从函数创建矩阵
    val matrix5: DenseMatrix[Int] = DenseMatrix.tabulate(3, 2){ case (i, j) => i + j}
    println(matrix5)
    /**
      * 输出结果
      * 0  1
      * 1  2
      * 2  3
      */

    // 从数组创建向量
    val vector8: DenseVector[Int] = new DenseVector(Array(1, 2, 3, 4))
    println(vector8)   // DenseVector(1, 2, 3, 4)

    // 从数组创建矩阵（2行3列）
    val matrix6: DenseMatrix[Int] = new DenseMatrix(2, 3, Array(11, 12, 13, 21, 22, 23))
    println(matrix6)
    /**
      * 输出结果
      * 11  13  22
      * 12  21  23
      */

    // 0-1 的随机向量 长度为4
    val vector9: DenseVector[Double] = DenseVector.rand[Double](4)
    println(vector9)  // DenseVector(0.695247172343499, 0.4823802019987977, 0.058256398502076534, 0.4210296409451917)

    // 0-1 的随机矩阵 2行3列
    val matrix7: DenseMatrix[Double] = DenseMatrix.rand[Double](2, 3)
    println(matrix7)
    /**
      * 0-1 的随机矩阵
      * 0.38565462721497035  0.03798386401540377  0.45663984553553494
      * 0.1567627501481168   0.15745804746791792  0.2534226309690739
      */

  }
}
