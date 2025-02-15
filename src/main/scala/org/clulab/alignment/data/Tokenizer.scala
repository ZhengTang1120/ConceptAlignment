package org.clulab.alignment.data

class Tokenizer {

  def tokenize(text: String): Array[String] = {
    text
        .split(' ')
        .filter(!_.isEmpty)
        .map(_.toLowerCase)
  }
}

object Tokenizer {

  def apply(): Tokenizer = new Tokenizer()
}
