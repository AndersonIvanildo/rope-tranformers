import numpy as np
import streamlit as st

from charts import attention_chart, rotation_chart
from rope import MAX_CHARS, METHODS, attention, frequencies, tokenize

st.set_page_config(page_title="RoPE | Posição na atenção", layout="wide")
st.title("Posição na atenção")
st.write("Compare RoPE, posição fixa senoidal e atenção sem informação de posição.")
text = st.text_area(
    "Frase", "O gato viu o gato na janela.", max_chars=MAX_CHARS,
    help="Palavras e pontuação são tokens separados. Limite de 48 tokens.", key="text",
)
try:
    tokens = tokenize(text)
except ValueError as exc:
    st.info(str(exc))
    st.stop()

with st.sidebar:
    st.header("Parâmetros")
    dim = st.selectbox("Dimensão da cabeça", [8, 16, 32, 64], index=1, key="dim")
    offset = st.slider("Posição inicial", 0, 100, 0, key="offset")
    causal = st.checkbox("Máscara causal", key="causal", help="Cada token só acessa a si mesmo e aos anteriores.")
    st.caption("Pesos e embeddings sintéticos, fixos e sem treino. Os mapas mostram o cálculo da atenção, sem avaliar compreensão da frase.")

positions = np.arange(len(tokens)) + offset
labels = [f"{p}: {token}" for p, token in zip(positions, tokens)]
results = attention(tokens, dim, offset, causal)
st.caption(f"{len(tokens)} tokens · {dim} dimensões · posições {positions[0]} a {positions[-1]}")
geometry, maps, values = st.tabs(["Rotação em 3D", "Mapas de atenção", "Valores e fórmulas"])

with geometry:
    controls, plot = st.columns([1, 3])
    with controls:
        selected = st.selectbox("Token observado", range(len(tokens)), format_func=lambda i: labels[i], key="token")
        pair = st.selectbox("Par de dimensões de Q", range(dim // 2), format_func=lambda i: f"{2 * i} e {2 * i + 1}", key="pair")
        frequency = frequencies(dim)[pair]
        st.metric("Rotação por posição", f"{np.degrees(frequency):.4f}°")
        st.write("O mesmo token é colocado em cada posição da frase. Assim, o gráfico isola o efeito da posição.")
        st.caption("O eixo vertical é a posição. Os outros dois eixos são um par de coordenadas de Q, sem PCA.")
    with plot:
        st.plotly_chart(rotation_chart(tokens[selected], dim, pair, positions), width="stretch", key="rotation")
    st.caption("RoPE gira cada par de Q e K e preserva seu comprimento. A posição fixa soma senos e cossenos ao embedding antes das projeções. Arraste o gráfico para mudar o ângulo de visão.")
    st.info("Experimente repetir uma palavra em várias posições. Depois altere a posição inicial e compare os mapas de atenção.")

with maps:
    st.write("Cada linha distribui a atenção de uma consulta entre as chaves. A soma da linha é 1. A escala de cor é a mesma nos três mapas, de 0 a 1.")
    for column, method in zip(st.columns(3), METHODS):
        with column:
            st.subheader(method)
            st.plotly_chart(attention_chart(results[method]["weights"], labels), width="stretch", key=method)
    st.caption("Atenção é softmax(QKᵀ / √d). Com a máscara causal, as posições futuras recebem peso zero.")
    shifted = attention(tokens, dim, offset + 10, causal)
    st.write("Efeito de deslocar todas as posições em +10, mantendo os tokens e os pesos:")
    for column, method in zip(st.columns(3), METHODS):
        difference = np.max(np.abs(results[method]["weights"] - shifted[method]["weights"]))
        column.metric(method, f"{difference:.2e}", help="Maior diferença absoluta entre os pesos de atenção antes e depois do deslocamento.")
    st.caption("Em RoPE, os produtos Q·K dependem da diferença entre posições: o deslocamento conjunto preserva a atenção, salvo arredondamento numérico. Sem codificação, os vetores dependem só do conteúdo; a máscara causal ainda restringe o acesso pela ordem. A soma de posição fixa com conteúdo pode mudar a atenção.")

with values:
    query = st.selectbox("Consulta para inspecionar", range(len(tokens)), format_func=lambda i: labels[i], key="query")
    st.dataframe(
        {"Token (chave)": labels, **{method: results[method]["weights"][query] for method in METHODS}},
        hide_index=True, width="stretch",
    )
    method = st.selectbox("Método para ver Q, K, V e saída", METHODS, index=2)
    tensor = st.selectbox("Matriz", ["q", "k", "v", "scores", "output"])
    st.dataframe(results[method][tensor], width="stretch")
    st.caption("Linhas seguem a ordem dos tokens. scores são os produtos escalados antes da máscara; output é a saída da cabeça, atenção × V.")
    st.latex(r"\theta_i=10000^{-2i/d},\quad \operatorname{PE}(p)_{2i}=\sin(p\theta_i),\quad \operatorname{PE}(p)_{2i+1}=\cos(p\theta_i)")
    st.latex(r"R(p\theta)\begin{bmatrix}a\\b\end{bmatrix}=\begin{bmatrix}a\cos(p\theta)-b\sin(p\theta)\\a\sin(p\theta)+b\cos(p\theta)\end{bmatrix}")
    st.latex(r"(R_m q)^\top(R_n k)=q^\top R_{n-m}k")
    st.write("Sem posição: X é projetado em Q, K e V. Fixa senoidal: X + PE é projetado em Q, K e V. RoPE: X é projetado e a rotação é aplicada somente a Q e K; V permanece igual ao caso sem posição.")
    st.caption("Aqui, posição fixa significa a fórmula senoidal do Transformer original. Embeddings posicionais aprendidos são outra alternativa: uma tabela de vetores ajustada no treino. Esta demonstração não faz treinamento.")
    st.markdown("Fontes: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) e [RoFormer](https://arxiv.org/abs/2104.09864).")
