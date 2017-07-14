package com.xj.ml.dataformats

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object SaveAsLibSVMFileDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LoadLibSVMFileDemo")
    val sc = new SparkContext(conf)

    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    val take: Array[LabeledPoint] = data.take(2)

    println("take:" + take)

    val rdd: RDD[LabeledPoint] = sc.parallelize(take)
    /**
      * 将LIBSVM格式的数据保存到指定的文件中
      * 传入参数 data(LabelPoint格式的RDD数据)
      * dir保存路径
      */

    // 输出路径需要不存在
    MLUtils.saveAsLibSVMFile(rdd, "D:\\IdeaWorkSpace\\SparkOperatorDemo\\output\\saveAsLibSVMFile")
  }
}
