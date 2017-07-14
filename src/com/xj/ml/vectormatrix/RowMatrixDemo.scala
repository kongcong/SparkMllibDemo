package com.xj.ml.vectormatrix

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix}
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/10
  */
object RowMatrixDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("RowMatrixDemo")
    val sc = new SparkContext(conf)

    val rdd: RDD[Array[Double]] = sc.parallelize(
      Array(
        Array(1.0, 2.0, 3.0, 4.0),
        Array(2.0, 3.0, 4.0, 5.0),
        Array(3.0, 4.0, 5.0, 6.0)
      ))
    val rdd1: RDD[linalg.Vector] = rdd.map(f => Vectors.dense(f))
    println("************** rdd1 *************")
    rdd1.foreach(println(_))
    /**
      * [1.0,2.0,3.0,4.0]
      * [2.0,3.0,4.0,5.0]
      * [3.0,4.0,5.0,6.0]
      */

    // 创建实例
    val RM: RowMatrix = new RowMatrix(rdd1)
    println("************** RM.rows *************")
    RM.rows.foreach(println(_))
    /**
      * [1.0,2.0,3.0,4.0]
      * [2.0,3.0,4.0,5.0]
      * [3.0,4.0,5.0,6.0]
      */

    // 计算每列之间的相似度(抽样)
    val simic1: CoordinateMatrix = RM.columnSimilarities(0.5)
    println("************** simic1 *************")
    simic1.entries.foreach(println(_))
    /**
      * MatrixEntry(2,3,0.7213475204444817)
      * MatrixEntry(0,1,1.0151337142356327)
      * MatrixEntry(1,2,1.1541560327111708)
      * MatrixEntry(0,3,0.5075668571178164)
      * MatrixEntry(1,3,0.829549648511154)
      * MatrixEntry(0,2,1.3196738285063225)
      */

    // 计算每列之间的相似度
    val simic2: CoordinateMatrix = RM.columnSimilarities()
    println("************** simic2 *************")
    simic2.entries.foreach(println(_))
    /**
      * MatrixEntry(2,3,0.9992204753914715)
      * MatrixEntry(0,1,0.9925833339709303)
      * MatrixEntry(1,2,0.9979288897338914)
      * MatrixEntry(0,3,0.9746318461970762)
      * MatrixEntry(1,3,0.9946115458726394)
      * MatrixEntry(0,2,0.9827076298239907)
      */

    // 计算每列的统计汇总
    val simic3: MultivariateStatisticalSummary = RM.computeColumnSummaryStatistics()
    println("********* computeColumnSummaryStatistics *********")
    println(simic3.max)     // [3.0,4.0,5.0,6.0]
    println(simic3.min)     // [1.0,2.0,3.0,4.0]
    println(simic3.mean)    // [2.0,3.0,4.0,5.0]

    // 计算每列之间的协方差，生成协方差矩阵
    val cc1: Matrix = RM.computeCovariance()
    println("********* computeCovariance *********")
    println(cc1)
    /**
      * 1.0  1.0  1.0  1.0
      * 1.0  1.0  1.0  1.0
      * 1.0  1.0  1.0  1.0
      * 1.0  1.0  1.0  1.0
      */

    /**
      * 计算格拉姆矩阵
      *
      * 给定一个实矩阵 A，
      * 矩阵 ATA 是 A 的列向量的格拉姆矩阵，
      * 而矩阵 AAT 是 A 的行向量的格拉姆矩阵。
      *
      */
    val cg1: Matrix = RM.computeGramianMatrix()
    println("********* computeGramianMatrix *********")
    println(cg1)
    /**
      * 14.0  20.0  26.0  32.0
      * 20.0  29.0  38.0  47.0
      * 26.0  38.0  50.0  62.0
      * 32.0  47.0  62.0  77.0
      */

    /**
      * 主成分分析计算
      * 取前k个主要变量，其结果矩阵的行为样本，列为变量
      */
    val cpc1: Matrix = RM.computePrincipalComponents(3)
    println("********* computePrincipalComponents *********")
    println(cpc1)
    /**
      * -0.5000000000000002  0.8660254037844388    1.6653345369377348E-16
      * -0.5000000000000002  -0.28867513459481275  0.8164965809277258
      * -0.5000000000000002  -0.28867513459481287  -0.40824829046386296
      * -0.5000000000000002  -0.28867513459481287  -0.40824829046386296
      */

    // 计算矩阵的奇异值分解
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = RM.computeSVD(4, true)
    println("********* SVD *********")
    svd.U.rows.foreach(println(_))
    /**
      * [-0.4176729387450486,0.8117158675136333,3.3527612686157227E-8,3.3527612686157227E-8]
      * [-0.5647271138038177,0.12006923114663204,1.1175870895385742E-8,0.0]
      * [-0.7117812888625867,-0.5715774052203693,-7.450580596923828E-9,-2.9802322387695312E-8]
      *
      */
    println("svd.s"+ svd.s)   // [13.011193721236575,0.8419251442105343,7.793650306633694E-8,5.761418127495863E-8]
    println(svd.V)
    /**
      * -0.2830233037672786  -0.7873358937103356  -0.5230588083704528  0.1625099473450276
      * -0.4132328277901395  -0.3594977469144485  0.5762839813994665   -0.6065449470421622
      * -0.5434423518130005  0.06834039988143598  0.41660846231241616  0.7255600520492322
      * -0.6736518758358616  0.4961785466773299   -0.4698336353414315  -0.28152505235209946
      */

    val B: Matrix = Matrices.dense(4, 3, Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    val multiply: RowMatrix = RM.multiply(B)
    multiply.rows.foreach(println(_))
    /**
      * [10.0,10.0,10.0]
      * [14.0,14.0,14.0]
      * [18.0,18.0,18.0]
      */

    val cols: Long = RM.numCols()
    println(cols)    // 4

    val rows: Long = RM.numRows()
    println(rows)    // 3

    // 矩阵转化成RDD
    RM.rows.foreach(println(_))
    /**
      * [1.0,2.0,3.0,4.0]
      * [2.0,3.0,4.0,5.0]
      * [3.0,4.0,5.0,6.0]
      */








  }
}
