package com.xj.ml.regression

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/10
  */
object LogisticRegressionDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegressionDemo")
    val sc = new SparkContext(conf)

    // 读取样本数据，为LibSVM格式 数据格式为：  标签 特征ID:特征值
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    // 样本数据划分训练样本和测试样本
    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training: RDD[LabeledPoint] = splits(0).cache()
    val test: RDD[LabeledPoint] = splits(1)

    // 新建逻辑回归模型，并训练
    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training)

    // 对测试样本进行测试
    val predictionAndLabels: RDD[(Double, Double)] = test.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val print_predict: Array[(Double, Double)] = predictionAndLabels.take(20)
    println("prediction" + "\t\t\t\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t\t\t" + print_predict(i)._2)
    }
    /**
      * prediction	label
      * 1.0			    1.0
      * 1.0		    	1.0
      * 0.0		    	0.0
      * 1.0		    	1.0
      * 0.0		    	0.0
      * 0.0		    	0.0
      * 1.0		    	1.0
      * 1.0			    1.0
      * 1.0	    		1.0
      * 0.0	    		0.0
      * 1.0			    1.0
      * 1.0   			1.0
      * 0.0		    	0.0
      * 0.0		    	1.0
      * 0.0	    		0.0
      * 0.0			    0.0
      * 1.0	    		1.0
      * 1.0		    	1.0
      * 1.0		    	1.0
      * 1.0	    		1.0
      */

    // 误差计算
    val metrics: MulticlassMetrics = new MulticlassMetrics(predictionAndLabels)
    val precision: Double = metrics.precision
    println("Precision = " + precision)  // Precision = 0.9705882352941176

    // 保存模型
    val modelPath = "modelpath/logisticRegressionModel"
    model.save(sc, modelPath)

    // 加载模型
    val load: LogisticRegressionModel = LogisticRegressionModel.load(sc, modelPath)

  }
}
