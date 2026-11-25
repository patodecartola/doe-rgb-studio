# DoE RGB Studio

Leitura colorimétrica de amostras em tubos de ensaio a partir de fotografias, com curva de calibração e exportação para planilha.

Substitui a leitura em espectrofotômetro em ensaios de triagem onde a diferença de cor é visível, usando uma foto comum como fonte do sinal.

## Como funciona

A absorbância é estimada diretamente da imagem:

```
A = -log10(sinal do tubo / branco de referência)
```

O branco de referência é uma região clara da foto, detectada automaticamente ou marcada com o botão direito. O sinal do tubo é a cor média dentro da ROI desenhada sobre ele.

Com dois ou mais pontos de concentração conhecida, o programa ajusta uma regressão linear e calcula R², S(y/x), erro padrão da inclinação, LOD e LOQ (3,3 e 10 vezes S(y/x) dividido pela inclinação).

## Fluxo de trabalho

O programa tem três abas, na ordem de uso:

1. **Planejamento** — abre um projeto do Minitab (`.mpx`) ou uma planilha CSV/XLSX, identifica os fatores do delineamento e a coluna de identificação dos ensaios, e aceita a concentração de cada padrão.
2. **Fotos e ROIs** — abre as fotos, detecta automaticamente os tubos e propõe uma ROI por tubo. As ROIs são polígonos editáveis: mover, redimensionar pela roda do mouse, arrastar vértices, adicionar e remover pontos, com desfazer e refazer.
3. **Sinal e calibração** — consolida as leituras, monta a curva de calibração, prevê a concentração das amostras e exporta tudo em XLSX com gráfico de dispersão e linha de tendência.

## Detalhes técnicos

- A detecção automática de tubos trabalha no espaço CIELAB, isolando regiões de croma destoante do fundo por transformada de distância e componentes conexos, com duas passagens de limites de tamanho (tubos lado a lado em bancada, ou foto aproximada de poucos tubos).
- Leitor próprio de projetos `.mpx` do Minitab, que são arquivos ZIP com metadados em JSON, incluindo detecção do separador decimal usado no projeto.
- Onze respostas disponíveis por ROI: A_R, A_G, A_B, norma de A, R, G, B, H, S, V e intensidade.
- Suporte a brancos de reagente (ensaios B1, B2, ...), com desconto da média.
- Sessões salvas em JSON, com compatibilidade retroativa para ROIs circulares de versões anteriores.
- Avisos automáticos de R² baixo, resposta não monotônica, IDs repetidos e fotos não conferidas.

## Instalação

```bash
pip install -r requirements.txt
python doe_rgb_studio.py
```

Requer Python 3.10 ou superior.

## Autor

Kauê Vinicius Romano da Luz — CTISM/UFSM
