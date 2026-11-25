# RoPE na atenção de um Transformer

Demonstração pequena de Rotary Position Embedding com NumPy, Streamlit e Plotly. Digite uma frase para comparar atenção sem posição, posição fixa senoidal e RoPE.

O app mostra a trajetória 3D de um mesmo token em posições diferentes, mapas de atenção com a mesma escala, os valores de Q, K, V e a saída da cabeça. É possível mudar a dimensão, o par de coordenadas, a posição inicial e a máscara causal.

Os embeddings e as projeções são sintéticos, determinísticos e sem treino. Palavras iguais, desconsiderando maiúsculas, têm o mesmo embedding. Os mapas ilustram o cálculo; não são explicações semânticas nem uma avaliação da qualidade de RoPE. O projeto implementa uma cabeça de atenção, sem o restante das camadas de um Transformer.

## Executar com uv

Use Python 3.11.5 ou posterior da série 3.11 e [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv run streamlit run app.py
```

Para usar exatamente as versões exportadas para o Cloud:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Notebook

```bash
uv sync --group notebook
uv run --group notebook jupyter lab demo_rope.ipynb
```

O [notebook](demo_rope.ipynb) usa os mesmos cálculos e gráficos do app. Tem as fórmulas, a comparação visual e experimentos sobre posição relativa, norma e máscara causal. Altere a frase e os parâmetros na primeira célula de código e execute as células em ordem.

## Publicar no Streamlit Community Cloud

Na criação do app em [share.streamlit.io](https://share.streamlit.io), use:

| Campo | Valor |
| :--- | :--- |
| Repository | `AndersonIvanildo/rope-tranformers` |
| Branch | A branch para a qual você enviou os arquivos |
| Main file path | `app.py` |
| Python, em Advanced settings | `3.11` |

O `requirements.txt` contém as dependências do app com versões fixadas, sem Jupyter. Não são necessários secrets, GPU, pesos de modelos, API externa ou `packages.txt`. Depois de instaladas as dependências, o cálculo funciona sem acesso à internet.

O texto aceita até 2.000 caracteres e 48 tokens, incluindo pontuação. A dimensão máxima é 64. As matrizes são pequenas e não há treinamento ou armazenamento das frases em arquivos. Os limites de recursos da hospedagem podem mudar; este projeto evita downloads e cálculos grandes, mas não promete uma cota específica.

O `uv.lock` local fica ignorado de propósito: o Community Cloud prioriza esse arquivo quando ele está junto ao `requirements.txt`. Assim, a hospedagem usa a exportação do app. O `pyproject.toml` mantém o grupo opcional `notebook` para desenvolvimento.

Após mudar dependências, atualize a exportação antes de enviar os arquivos:

```bash
uv lock
uv export --no-dev --no-group notebook --no-emit-project --no-hashes --no-annotate --output-file requirements.txt
```

O `requirements.txt` foi exportado para Linux, o ambiente do Community Cloud. Suas linhas incluem o marcador `sys_platform == 'linux'`. Em outros sistemas, use `uv sync`.

## Verificação

```bash
uv run python -m unittest discover -s tests -v
```

Os testes cobrem as fórmulas, preservação da norma, invariância ao deslocar posições com RoPE, normalização e máscara causal, entradas inválidas e alterações dos controles no app.

## O que observar

1. Digite `gato gato gato gato`. Sem posição e sem máscara causal, a atenção é uniforme. RoPE diferencia as posições pela rotação de Q e K.
2. No gráfico 3D, altere o par de dimensões. Cada par usa uma frequência diferente. O eixo vertical representa a posição, não uma dimensão extra do embedding.
3. Nos mapas, altere a posição inicial. O conteúdo permanece igual. A atenção com RoPE é preservada, salvo arredondamento, porque o produto entre Q e K depende da diferença entre posições. Com a soma senoidal, o conteúdo interage com posições absolutas.
4. Ative a máscara causal. Os pesos para tokens futuros ficam em zero.

RoPE não garante que a atenção diminua sempre com a distância, nem que um modelo extrapole corretamente para qualquer contexto. Embeddings posicionais aprendidos são outra alternativa, descrita no notebook; não há treinamento neste projeto.

## Referências

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Dependências no Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Publicação e versão de Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
