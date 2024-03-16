from behave import *

use_step_matcher("re")


@given("J'ai Pierrot de type (.+)")
def step_impl(context, arg0):
    """
    :type context: behave.runner.Context
    :type arg0: str
    """
    # on fait rien mais c'est pas un vrai test c'est pour montrer


@when("Je lis son blase à voix haute")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    # on fait rien
    # mais c'est un peu plus "normal"
    # t'façons même dans les vrais tests j'ai pas trop tendance à employer le when de toutes façons ...


@then("je me sens (?P<sentiment>.+)")
def step_impl(context, sentiment):
    """
    :type context: behave.runner.Context
    :type sentiment: str
    """
    # non mais on fait VRAIMENT rien j'allais en plus inventer des assert ?
    # ça va là le poc il est fait et puis j'ai pas d'env de dev python qui tourne flemme d'aller plus loin xD
