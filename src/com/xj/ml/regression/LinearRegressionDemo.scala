package com.xj.ml.regression

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/10
  */
object LinearRegressionDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LinearRegressionDemo")
    val sc = new SparkContext(conf)

    /**
      * 读取样本数据
      *
      * 认识两种数据
      * 1、普通标签数据，数据格式为："标签，特征值1 特征值2 特征值3...."
      * 2、LibSVM格式的数据，数据格式为："标签 特征ID:特征值 ...."
      *
      */
    val file: RDD[String] = sc.textFile("lpsa.data")
    val examples: RDD[LabeledPoint] = file.map { line =>
      // 使用逗号切分出标签和所有特征，使用空格切分各个特征
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()

    //val file: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_linear_regression_data.txt").cache()
    val numExamples: Long = file.count()

    /**
      * 新建线性回归模型，并设置训练参数
      */
    val model: LinearRegressionModel = LinearRegressionWithSGD.train(examples, 100, 1, 1.0)

    /**
      * 获取权重
      */
    val weights: linalg.Vector = model.weights
    println(s"weights: $weights")
    /**
      * weights: [0.5523924782328413,0.2124029986475655,0.3433043481387762,0.14519087637306177,
      * 0.35272849296086883,-0.6151213105429235,-0.5997911631155013,0.8631706297594202]
      */

    /**
      * 获取偏置项
      */
    val intercept: Double = model.intercept
    println(s"intercept: $intercept")
    /**
      * intercept: 0.0
      */

    /**
      * 对样本进行测试
      */
    val predict: RDD[Double] = model.predict(examples.map(_.features))
    val predictionAndLable: RDD[(Double, Double)] = predict.zip(examples.map(_.label))
    val print_predict: Array[(Double, Double)] = predictionAndLable.take(50)
    println("prediction" + "\t\t\t\t" + "label")
    for (i <- 0 to print_predict.length - 1){
      println(print_predict(i)._1 + "\t\t" + print_predict(i)._2)
    }
    /**
      * prediction				    label
      * -1.8933517267847382		-0.4307829
      * -1.4459273206933823		-0.1625189
      * -1.0116143535344557		-0.1625189
      * -1.5624816747894203		-0.1625189
      * -0.387431044790143		0.3715636
      * -1.88588023108536		  0.7654678
      * -0.25353019228791596	0.8544153
      * -0.3967629363642373		1.2669476
      * -0.9768502193872691		1.2669476
      * -0.3996541161284979		1.2669476
      * -0.7095922968298545		1.3480731
      * -0.016950669349060754	1.446919
      * -0.5200457885339715		1.4701758
      * -0.14833980679997827	1.4929041
      * -2.053051254487376		1.5581446
      * ....
      */

    /**
      * 计算测试误差
      *
      */
    val valuesAndPreds = examples.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)
    // training Mean Squared Error = 6.207597210613579

    // 模型保存
    val modelPath = "modelpath"
    model.save(sc, modelPath)

    // 加载模型
    val load: LinearRegressionModel = LinearRegressionModel.load(sc, modelPath)

  }
}
