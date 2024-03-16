Feature: Pierrot
  pour tester des trucs

  Scenario Outline: Pierrot à plusieurs alias mais je l'es aime plus ou moins
    Given J'ai Pierrot de type <type de Pierrot>
    When Je lis son blase à voix haute
    Then je me sens <sentiment>

    Examples: les noms qui me vont (sans particulièrement que j'y tienne non plus spécialement)
      | type de Pierrot | sentiment                                             |
      | l'asticot       | amusé parce que je trouve que c'est rigolo et ça rime |

    Examples: les appellations avec lesquelles je suis moins à l'aise
      | type de Pierrot | sentiment                                                                                                      |
      | Le Con          | neutre parce que c'est lui qui l'a choisi et que, dans le fond, le principal c'est que ça lui plaise à lui ... |
      | le viewer       | circonspect parce que je trouve ça inélégant de le réduir à cette qualité extrinsèque                          |

    Examples: les blases jamais vus
      | type de Pierrot | sentiment                                                                                                                        |
      | l'ornithorynque | surpris d'ouf parce que je savais pas qu'on l'appelait comme ça ... et en vrai je crois que c'est pas du tout sa dénomination ... |
