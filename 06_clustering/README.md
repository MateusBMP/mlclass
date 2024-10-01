Utilizar o arquivo anexo para encontrar os perfis de olhos (clustering) baseado nas respectivas medidas:
AL = comprimento axial do olho
ACD = profundidade de câmara anterior
WTW = distância brando a branco
K1 = curvatura no meridiano menos curvo
K2 = curvatura no meridiano mais curvo

A atividade será apresentar o perfil dos grupos encontrados, assim foquem em uma apresentação como se fosse para um cliente de vocês. Alguém que contratou o seu s serviços de Engenheiro de Conhecimento, especialista em aprendizagem de máquina, data mining, business intelligence... o nome que vocês quiserem e acharem mais bonito :-)

Atenção: a atividade consiste nos grupos encontrados baseados somente nas variáveis acima. Somente depois dos grupos definidos é vocês se quiserem complementar a apresentação de vocês podem acrescentar  a frequência da coluna correto para cada grupo. Essa é só um complemento para a descrição dos grupos no momento da apresentação não é uma variável a ser considerada para se identificar o padrão de olhos presentes na base. Cuidado para não usar isso e confundir a ideia de grupo com classe!

Se quiserem entender melhor o que corresponde cada variável listada acima fisicamente/anatomicamente no olho podem usar a referência:
https://www.draandreia.com.br/?page_id=1027
https://www.draandreia.com.br/wp-content/uploads/2017/09/biom4.png

## Informações

Para calculcar a Dioptria do cristalino, utilizamos a equação de Lensmaker:

> _D = (n - 1) * (1/R1 - 1/R2)_

Sendo a curvatura _C = 1/R_, equivalentes a K1 e K2, e _n = 1.376_, equivalente ao índice de refração da córnea. Assim, temos a seguinte equação para a dioptria do cristalino:

> _D = 0.376 * (K2 - K1)_