package org.clulab.alignment.controllers.v1

import java.util.concurrent.TimeUnit

import javax.inject._
import org.clulab.alignment.Locations
import org.clulab.alignment.SingleKnnApp
import org.clulab.alignment.SingleKnnAppTrait
import org.clulab.alignment.controllers.utils.Busy
import org.clulab.alignment.controllers.utils.Ready
import org.clulab.alignment.controllers.utils.StatusHolder
import org.clulab.alignment.searcher.lucene.document.DatamartDocument
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import play.api.mvc._
import play.api.libs.json.Json
import play.api.libs.json.JsArray
import play.api.libs.json.JsValue
import play.api.mvc.Action

import scala.concurrent.Await
import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.concurrent.duration.FiniteDuration

class SingleKnnAppFuture(statusHolder: StatusHolder) extends SingleKnnAppTrait {
  import scala.concurrent.ExecutionContext.Implicits.global

  protected val singleKnnAppFuture: Future[SingleKnnApp] = Future {
    val result = new SingleKnnApp()
    statusHolder.set(Ready)
    result
  }

  override def run(queryString: String, maxHits: Int): Seq[(DatamartDocument, Float)] = {
    val maxWaitTime: FiniteDuration = Duration(200, TimeUnit.SECONDS)
    val runFuture = singleKnnAppFuture.map { singleKnnApp =>
      singleKnnApp.run(queryString, maxHits)
    }

    Await.result(runFuture, maxWaitTime)
  }
}

@Singleton
class HomeController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {
  import HomeController.logger

  val statusHolder: StatusHolder = new StatusHolder(logger, Busy)
  val maxMaxHits = 500

  {
    println("Configuring...")

    def configure(filename: String): Unit = {
      val canonicalPath = new java.io.File(filename).getCanonicalPath
      println("Place file here: " + canonicalPath)
    }

    configure(Locations.datamartFilename)
    configure(Locations.luceneDirname)
    configure(Locations.gloveFilename)
  }

  println("Initializing...")
  val singleKnnAppFuture = new SingleKnnAppFuture(statusHolder)

  println("Up and running...")

  def index(): Action[AnyContent] = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index())
  }

  def ping: Action[AnyContent] = Action {
    logger.info("Called 'ping' function!")
    Ok
  }

  def echo(text: String): Action[AnyContent] = Action {
    logger.info(s"Called 'echo' function with '$text'!")
    Ok(text)
  }

  def status: Action[AnyContent] = Action {
    logger.info("Called 'status' function!")
    Ok(statusHolder.toJsValue)
  }

  def search(query: String, maxHits: Int): Action[AnyContent] = Action {
    logger.info(s"Called 'search' function with '$query' and '$maxHits'!")
    val hits = math.min(maxMaxHits, maxHits) // Cap it off at some reasonable amount.
    val datamartDocumentsAndScores: Seq[(DatamartDocument, Float)] = singleKnnAppFuture.run(query, hits)
    val jsObjects = datamartDocumentsAndScores.map { case (datamartDocument, score) =>
      Json.obj(
      "score" -> score,
        "datamartId" -> datamartDocument.datamartId,
        "datasetId" -> datamartDocument.datasetId,
        "variableId" -> datamartDocument.variableId,
        "variableName" -> datamartDocument.variableName,
        "variableDescription" -> datamartDocument.variableDescription
      )
    }
    val jsValue: JsValue = JsArray(jsObjects)

    Ok(jsValue)
  }
}

object HomeController {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
}
