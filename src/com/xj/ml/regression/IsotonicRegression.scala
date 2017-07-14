package com.xj.ml.regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/10
  */
object IsotonicRegressionDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("IsotonicRegression")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    /**
      * 读取样本数据
      *
      * 普通标签格式数据 数据格式为 "标签，特征值"
      * 例如：
      * 0.24579296,0.01
      * 0.28505864,0.02
      * 0.31208567,0.03
      * 0.35900051,0.04
      * 0.35747068,0.05
      * 0.16675166,0.06
      *
      */
    val data: RDD[String] = sc.textFile("sample_isotonic_regression_data.txt")
    val parseData: RDD[(Double, Double, Double)] = data.map { line =>
      val parts = line.split(",").map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }

    /**
      * 样本数据划分训练样本与测试样本
      */
    val splits: Array[RDD[(Double, Double, Double)]] = parseData.randomSplit(Array(0.6, 0.4), seed = 12L)
    val training: RDD[(Double, Double, Double)] = splits(0)
    val test: RDD[(Double, Double, Double)] = splits(1)

    /**
      * 新建保序回归模型并训练
      *
      * boundaries: 边界数组，即分段函数X的分段点数组，边界数组按顺序存储
      * predictions: 对应边界数组的y值，即分段函数x的分段点对应的Y值
      *
      */
    val model: IsotonicRegressionModel = new IsotonicRegression()  // 引用mllib包下的IsotonicRegression
      .setIsotonic(true)     // 设置升序、降序参数
      .run(training)         // 模型训练run方法
    // 取X
    val x: Array[Double] = model.boundaries
    // 取最终保序Y
    val y: Array[Double] = model.predictions
    println("boundaries" + "\t\t" + "predictions")
    for (i <- 0 to(x.length - 1))
      println(x(i) + "\t\t\t" + y(i))
    /**
      * boundaries		predictions
      * 0.01			0.16490801384615403
      * 0.17			0.16490801384615403
      * 0.18			0.19479857375000006
      * 0.27			0.19479857375000006
      * 0.28			0.20040796
      * 0.29			0.29576747
      * 0.3		  	0.43396226
      * 0.31			0.5081591025000001
      * 0.34			0.5081591025000001
      * 0.35			0.54156043
      * 0.36			0.5602243760000001
      * 0.41			0.5602243760000001
      * 0.44			0.567690657241378
      * 0.75			0.567690657241378
      * 0.76			0.57929628
      * 0.77			0.64762876
      */

    /**
      * 误差计算
      */
    val predictionAndLabel: RDD[(Double, Double)] = test.map { point =>
      val predictedLabel: Double = model.predict(point._2)
      (predictedLabel, point._1)
    }
    val print_predict: Array[(Double, Double)] = predictionAndLabel.take(20)
    println("prediction" + "\t" + "label")
    for (i <- 0 to print_predict.length - 1)
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    /**
      * prediction	label
      * 0.16868944399999988	0.31208567
      * 0.16868944399999988	0.35900051
      * 0.16868944399999988	0.03926568
      * 0.16868944399999988	0.12952575
      * 0.16868944399999988	0.0
      * 0.16868944399999988	0.01376849
      * 0.16868944399999988	0.13105558
      * 0.19545421571428565	0.13717491
      * 0.19545421571428565	0.19020908
      * 0.19545421571428565	0.19581846
      * 0.31718510999999966	0.29576747
      * 0.5322114566666667	0.4854666
      * 0.5368859433333334	0.49209587
      * 0.5602243760000001	0.5017848
      * 0.5701674724126985	0.58286588
      * 0.5801105688253968	0.64660887
      * 0.5900536652380952	0.65782764
      * 0.5900536652380952	0.63029067
      * 0.5900536652380952	0.63233044
      * 0.5900536652380952	0.33299337
      */
    val meanSquaredError: Double = predictionAndLabel.map {case(v, p) => math.pow((v - p), 2)}.mean()
    println("mean squared error = " + meanSquaredError)   // mean squared error = 0.006249261725472803

    /**
      * 保存模型
      */
    val modelPath = "modelpath/IsotonicRegressionModel"
    model.save(sc, modelPath)

    /**
      * 加载模型
      */
    val load: IsotonicRegressionModel = IsotonicRegressionModel.load(sc, modelPath)

  }
}
