package com.xj.ml.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object NaiveBayesDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("NaiveBayesDemo")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    /**
      * 读取样本数据
      * 数据格式为：类别，特征1 特征2 特征3
      * 例如：
      * 0,1 0 0
      * 0,2 0 0
      * 1,0 1 0
      * 1,0 2 0
      * 2,0 0 1
      * 2,0 0 2
      */
    val data: RDD[String] = sc.textFile("sample_naive_bayes_data.txt")
    val parsedData: RDD[LabeledPoint] = data.map { line =>
      val parts: Array[String] = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    /**
      * 将样本数据划分为训练样本与测试样本
      */
    val splits: Array[RDD[LabeledPoint]] = parsedData.randomSplit(Array(0.6, 0.4),seed = 11L)
    val training = splits(0)
    val test = splits(1)

    /**
      * 新建贝叶斯分类模型，并训练
      *
      * @param input 样本RDD,格式为RDD(label, features)
      * @param lambda 平滑参数
      * @param modelType 模型类型：多项式或者伯努利
      *                  multinomial 多项式 适合于训练集大到内存无法一次性放入的情况
      *                  bernoulli 伯努利 对于一个样本来说，其特征用的是全局的特征，每个特征的取值是布尔型
      */
    val model: NaiveBayesModel = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    /**
      * 对测试样本进行测试
      */
    val predictionAndLabel: RDD[(Double, Double)] = test.map(p => (model.predict(p.features), p.label))

    val print_predict: Array[(Double, Double)] = predictionAndLabel.take(20)
    println("prediction" + "\t\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }
    /**
      * prediction	label
      * 0.0	        0.0
      * 0.0	        0.0
      * 2.0	        2.0
      * 2.0	        2.0
      * 2.0	        2.0
      */

    // 计算贝叶斯分类精度
    val accuracy: Double = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println(s"accuracy = $accuracy")    // accuracy = 1.0

    // 保存与加载模型
    val modelPath = "modelpath/NaiveBayesModel"
    model.save(sc, modelPath)
    NaiveBayesModel.load(sc, modelPath)


  }
}
