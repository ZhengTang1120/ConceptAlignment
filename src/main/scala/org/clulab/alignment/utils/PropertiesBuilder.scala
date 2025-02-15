package org.clulab.alignment.utils

import java.util.Properties

import org.clulab.alignment.utils.Closer.AutoCloser

import scala.collection.JavaConverters._

class PropertiesBuilder(properties: Properties) {
  def put(key: AnyRef, value: AnyRef): PropertiesBuilder = {
    properties.put(key, value)
    this
  }

  def get: Properties = properties

  def getProperty(key: String): Option[String] = Option(properties.getProperty(key))

  def putAll(propertiesBuilder: PropertiesBuilder): PropertiesBuilder = {
    properties.putAll(propertiesBuilder.get)
    this
  }

  def filter(prefix: String): PropertiesBuilder = {
    val propertiesBuilder = PropertiesBuilder()
    val start = prefix + "."

    properties.asScala.foreach { case (key, value) =>
      if (key.startsWith(start))
        propertiesBuilder.put(key.drop(start.length), value)
    }

    propertiesBuilder
  }
}

object PropertiesBuilder {

  def apply(properties: Properties): PropertiesBuilder = new PropertiesBuilder(properties)

  def apply(): PropertiesBuilder = new PropertiesBuilder(new Properties)

  def fromResource(resourceName: String): PropertiesBuilder = {
    val properties = new Properties

    PropertiesBuilder.getClass.getResourceAsStream(resourceName).autoClose { inputStream =>
      properties.load(inputStream)
    }

    new PropertiesBuilder(properties)
  }

  def fromFile(fileName: String): PropertiesBuilder = {
    val properties = new Properties

    FileUtils.newBufferedInputStream(fileName).autoClose { bufferedInputStream =>
      properties.load(bufferedInputStream)
    }

    new PropertiesBuilder(properties)
  }
}
