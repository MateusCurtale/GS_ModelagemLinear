from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Define o arquivo CSV usado quando nenhum caminho e informado no terminal.
CAMINHO_PADRAO = Path(__file__).with_name("Space_Corrected.csv")

# Escolhe uma variavel quantitativa discreta: valores inteiros contaveis.
VARIAVEL_DISCRETA = "Anos"

# Escolhe uma variavel quantitativa continua: valor numerico que pode variar em escala decimal.
VARIAVEL_CONTINUA = "Cost"


def carregar_base(caminho_arquivo):
    """Le o arquivo CSV e devolve os dados em um DataFrame do pandas."""
    caminho_arquivo = Path(caminho_arquivo)

    # Interrompe o programa com uma mensagem clara se o arquivo nao existir.
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_arquivo}")

    return pd.read_csv(caminho_arquivo)


def validar_colunas(base, colunas):
    """Confere se as colunas escolhidas realmente existem na base."""
    colunas_faltantes = [coluna for coluna in colunas if coluna not in base.columns]

    # Mostra as colunas disponiveis para facilitar a correcao do nome.
    if colunas_faltantes:
        colunas_disponiveis = ", ".join(base.columns)
        raise ValueError(
            f"Coluna(s) nao encontrada(s): {colunas_faltantes}\n"
            f"Colunas disponiveis: {colunas_disponiveis}"
        )


def preparar_coluna_numerica(base, coluna):
    """Converte uma coluna para numero e remove valores vazios ou invalidos."""
    valores = base[coluna].astype(str).str.strip()
    valores_limpos = valores.str.replace(",", "", regex=False)
    dados_convertidos = pd.to_numeric(valores_limpos, errors="coerce")

    # Alguns registros de "Anos" aparecem como datas completas; nesses casos,
    # aproveita o ano contido no texto para nao perder observacoes da base.
    if coluna == VARIAVEL_DISCRETA:
        registros_invalidos = dados_convertidos.isna()
        anos_extraidos = valores.loc[registros_invalidos].str.extract(
            r"(\d{4})",
            expand=False,
        )
        dados_convertidos.loc[registros_invalidos] = pd.to_numeric(
            anos_extraidos,
            errors="coerce",
        )

    dados = dados_convertidos.dropna()

    if coluna == VARIAVEL_DISCRETA:
        dados = dados.astype(int)

    # A tabela de frequencia precisa de pelo menos um valor numerico valido.
    if dados.empty:
        raise ValueError(f"A coluna '{coluna}' nao possui valores numericos validos.")

    return dados


def formatar_numero(valor):
    """Formata numeros para exibir os resultados de forma mais limpa."""
    if pd.isna(valor):
        return "Indefinido"

    valor = float(valor)

    if valor.is_integer():
        return f"{valor:.0f}"

    return f"{valor:.2f}"


def formatar_moda(dados):
    """Formata a moda, considerando que uma variavel pode ter mais de uma moda."""
    modas = dados.mode()

    if modas.empty:
        return "Sem moda"

    valores_formatados = [formatar_numero(valor) for valor in modas]

    if len(valores_formatados) > 5:
        primeiros_valores = ", ".join(valores_formatados[:5])
        return f"{primeiros_valores} ... ({len(valores_formatados)} modas)"

    return ", ".join(valores_formatados)


def criar_tabela_estatistica_descritiva(dados):
    """Calcula medidas de tendencia central, dispersao e separatrizes."""
    media = dados.mean()
    mediana = dados.median()
    minimo = dados.min()
    maximo = dados.max()
    amplitude = maximo - minimo
    variancia = dados.var(ddof=0)
    desvio_padrao = dados.std(ddof=0)
    coeficiente_variacao = (
        (desvio_padrao / abs(media)) * 100
        if media != 0
        else float("nan")
    )
    quartis = dados.quantile([0.25, 0.5, 0.75])

    medidas = [
        ("Tendencia Central", "Media", media),
        ("Tendencia Central", "Mediana", mediana),
        ("Tendencia Central", "Moda", formatar_moda(dados)),
        ("Dispersao", "Minimo", minimo),
        ("Dispersao", "Maximo", maximo),
        ("Dispersao", "Amplitude", amplitude),
        ("Dispersao", "Variancia", variancia),
        ("Dispersao", "Desvio Padrao", desvio_padrao),
        ("Dispersao", "Coeficiente de Variacao (%)", coeficiente_variacao),
        ("Separatrizes", "1o Quartil (Q1)", quartis.loc[0.25]),
        ("Separatrizes", "2o Quartil (Q2)", quartis.loc[0.5]),
        ("Separatrizes", "3o Quartil (Q3)", quartis.loc[0.75]),
    ]

    valores_formatados = [
        formatar_numero(valor) if not isinstance(valor, str) else valor
        for _, _, valor in medidas
    ]

    return pd.DataFrame(
        {
            "Grupo": [grupo for grupo, _, _ in medidas],
            "Medida": [medida for _, medida, _ in medidas],
            "Valor": valores_formatados,
        }
    )


def preparar_dados_anos_cost(base):
    """Converte e alinha os dados de ano e custo para analise por ano."""
    base = base.copy()
    base[VARIAVEL_DISCRETA] = preparar_coluna_numerica(base, VARIAVEL_DISCRETA)
    base[VARIAVEL_CONTINUA] = preparar_coluna_numerica(base, VARIAVEL_CONTINUA)
    dados = base[[VARIAVEL_DISCRETA, VARIAVEL_CONTINUA]].dropna()

    if dados.empty:
        raise ValueError(
            "Nao ha dados validos suficientes para analisar a tendencia de preco por ano."
        )

    return dados


def criar_tabela_tendencia_ano(dados):
    """Cria uma tabela de medias de preco por ano e variacao anterior."""
    tabela = (
        dados.groupby(VARIAVEL_DISCRETA)[VARIAVEL_CONTINUA]
        .agg(Media="mean", Mediana="median", Minimo="min", Maximo="max", Contagem="count")
        .reset_index()
        .sort_values(VARIAVEL_DISCRETA)
    )

    tabela["Media"] = tabela["Media"].round(2)
    tabela["Mediana"] = tabela["Mediana"].round(2)
    tabela["Minimo"] = tabela["Minimo"].round(2)
    tabela["Maximo"] = tabela["Maximo"].round(2)
    tabela["Mudanca Media (%)"] = (
        tabela["Media"].pct_change().mul(100).round(2)
    )
    tabela["Mudanca Media (%)"] = tabela["Mudanca Media (%)"].fillna(0).map("{:+.2f}%".format)

    return tabela


def gerar_insights_tendencia_ano(dados):
    """Gera insights sobre a tendencia dos precos ao longo dos anos."""
    medias_ano = dados.groupby(VARIAVEL_DISCRETA)[VARIAVEL_CONTINUA].mean().sort_index()
    media_geral = dados[VARIAVEL_CONTINUA].mean()
    primeiro_ano = medias_ano.index[0]
    ultimo_ano = medias_ano.index[-1]
    primeiro_valor = medias_ano.iloc[0]
    ultimo_valor = medias_ano.iloc[-1]
    diferenca = ultimo_valor - primeiro_valor

    if primeiro_valor == 0:
        movimento = "nao e possivel calcular a variacao percentual"
    else:
        percentual = (diferenca / primeiro_valor) * 100
        movimento = (
            f"um aumento de {diferenca:.2f} ({percentual:+.2f}%)"
            if diferenca > 0
            else f"uma reducao de {abs(diferenca):.2f} ({percentual:+.2f}%)"
            if diferenca < 0
            else "nenhuma variacao significativa"
        )

    insights = [
        f"# Insight de Tendencia: a media de preco passou de {primeiro_valor:.2f} em {primeiro_ano} para {ultimo_valor:.2f} em {ultimo_ano}, representando {movimento}.",
        f"# Insight de Media: a media geral de preco no periodo analisado e {media_geral:.2f}.",
    ]

    return insights


def imprimir_insights(titulo, insights):
    """Exibe uma lista de insights de forma clara."""
    print("\n" + titulo)
    print("-" * len(titulo))
    for insight in insights:
        print(insight)
    print()


def plotar_tendencia_ano(tabela_tendencia):
    """Plota um grafico da media e mediana de preco por ano."""
    anos = tabela_tendencia[VARIAVEL_DISCRETA]
    medias = tabela_tendencia["Media"]
    medianas = tabela_tendencia["Mediana"]

    plt.figure(figsize=(10, 5))
    plt.plot(anos, medias, marker="o", linestyle="-", label="Media")
    plt.plot(anos, medianas, marker="s", linestyle="--", label="Mediana")
    plt.title("Tendencia de preco por ano")
    plt.xlabel("Ano")
    plt.ylabel("Preco (Milhão USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    filename = "tendencia_preco_ano.png"
    plt.savefig(filename)
    print(f"\nGrafico salvo em: {filename}")
    try:
        plt.show()
    except Exception:
        pass


def criar_tabela_discreta(base, coluna):
    """Monta a tabela de frequencia para a variavel quantitativa discreta."""
    dados = preparar_coluna_numerica(base, coluna)

    # Conta quantas vezes cada valor aparece e ordena do menor para o maior.
    tabela = (
        dados.value_counts()
        .sort_index()
        .rename_axis(coluna)
        .reset_index(name="fi")
    )

    # Calcula frequencia relativa, percentual e frequencias acumuladas.
    total = tabela["fi"].sum()
    tabela["fr"] = (tabela["fi"] / total).round(4)
    tabela["%"] = (tabela["fr"] * 100).round(2)
    tabela["Fi"] = tabela["fi"].cumsum()
    tabela["% acumulado"] = ((tabela["Fi"] / total) * 100).round(2)

    return tabela, dados


def calcular_quantidade_classes(total_registros):
    """Calcula a quantidade de classes pela regra de Sturges."""
    if total_registros <= 1:
        return 1

    return math.ceil(1 + 3.322 * math.log10(total_registros))


def formatar_intervalo(intervalo):
    """Deixa cada intervalo da tabela continua mais legivel."""
    return f"{intervalo.left:.2f} a {intervalo.right:.2f}"


def criar_tabela_continua(base, coluna):
    """Monta a tabela de frequencia para a variavel quantitativa continua."""
    dados = preparar_coluna_numerica(base, coluna)

    # A regra de Sturges cria uma quantidade equilibrada de classes.
    quantidade_classes = calcular_quantidade_classes(len(dados))

    # Agrupa os valores numericos em intervalos, pois variaveis continuas
    # costumam ter muitos valores diferentes.
    classes = pd.cut(dados, bins=quantidade_classes, include_lowest=True)

    # Conta os valores em cada intervalo e mantem a ordem natural das classes.
    tabela = (
        classes.value_counts()
        .sort_index()
        .rename_axis("Classe")
        .reset_index(name="fi")
    )

    # Calcula frequencia relativa, percentual e frequencias acumuladas.
    total = tabela["fi"].sum()
    tabela["Classe"] = tabela["Classe"].apply(formatar_intervalo)
    tabela["fr"] = (tabela["fi"] / total).round(4)
    tabela["%"] = (tabela["fr"] * 100).round(2)
    tabela["Fi"] = tabela["fi"].cumsum()
    tabela["% acumulado"] = ((tabela["Fi"] / total) * 100).round(2)

    return tabela, dados


def gerar_insights_discreta(tabela, dados, coluna):
    """Gera dois comentarios interpretando a tabela da variavel discreta."""
    maior_frequencia = tabela["fi"].max()
    valores_moda = tabela.loc[tabela["fi"] == maior_frequencia, coluna].tolist()
    percentual_moda = (maior_frequencia / len(dados)) * 100
    mediana = dados.median()
    # Insight 1: identifica o(s) valor(es) mais frequente(s) da variavel discreta.
    if len(valores_moda) == 1:
        insight_moda = (
            f"# Insight 1: O valor mais frequente de '{coluna}' é "
            f"{valores_moda[0]:.0f}, presente em {maior_frequencia} registros "
            f"({percentual_moda:.2f}% da base)."
        )
    else:
        valores_formatados = ", ".join(f"{valor:.0f}" for valor in valores_moda)
        insight_moda = (
            f"# Insight 1: Os valores mais frequentes de '{coluna}' são "
            f"{valores_formatados}, cada um com {maior_frequencia} registros "
            f"({percentual_moda:.2f}% da base)."
        )

    # Insight 2: usa a mediana para resumir a distribuicao dos valores discretos.
    insight_mediana = (
        f"# Insight 2: A mediana de '{coluna}' é {mediana:.0f}; isso indica que "
        f"aproximadamente metade dos registros ocorreu até {mediana:.0f}."
    )

    return [insight_moda, insight_mediana]


def gerar_insights_continua(tabela, dados, coluna):
    """Gera dois comentarios interpretando a tabela da variavel continua."""
    classe_mais_frequente = tabela.loc[tabela["fi"].idxmax()]
    media = dados.mean()
    mediana = dados.median()
    desvio_padrao = dados.std(ddof=0)
    # Insight 1: destaca a faixa de valores com maior concentracao de registros.
    insight_classe = (
        f"# Insight 1: A classe mais frequente de '{coluna}' é "
        f"{classe_mais_frequente['Classe']}, com {classe_mais_frequente['fi']} "
        f"registros ({classe_mais_frequente['%']:.2f}% da base)."
        f" O desvio padrao de {coluna} é {desvio_padrao:.3f}, indicando a variabilidade dos preços."
    )

    # Insight 2: compara media e mediana para observar a direcao da distribuicao.
    # A tolerancia evita interpretar como diferenca relevante valores quase iguais.
    diferenca = media - mediana
    if abs(diferenca) <= 0.005:
        interpretacao = (
            "a media e a mediana ficaram praticamente iguais, sugerindo uma "
            "distribuicao equilibrada de preços"
        )
    elif diferenca > 0:
        interpretacao = "a media ficou acima da mediana, sugerindo alguns valores mais altos"
    else:
        interpretacao = "a media ficou abaixo da mediana, sugerindo alguns valores mais baixos"

    insight_media = (
        f"# Insight 2: A media de '{coluna}' e {media:.3f}, enquanto a mediana "
        f"e {mediana:.3f}; {interpretacao}."
    )

    return [insight_classe, insight_media]


def imprimir_resultado(titulo, tabela, insights):
    """Exibe uma tabela de frequencia seguida dos insights em formato de comentario."""
    print("\n" + titulo)
    print("-" * len(titulo))
    print(tabela.to_string(index=False))
    print()

    # Os insights sao impressos com # para aparecerem como comentarios.
    for insight in insights:
        print(insight)


def imprimir_tabela(titulo, tabela):
    """Exibe uma tabela sem comentarios adicionais."""
    print("\n" + titulo)
    print("-" * len(titulo))
    print(tabela.to_string(index=False))


def main():
    """Executa o fluxo completo do programa."""
    # Se o usuario informar um caminho no terminal, usa esse arquivo.
    # Caso contrario, usa o CSV padrao que esta na mesma pasta do script.
    caminho_base = Path(sys.argv[1]) if len(sys.argv) > 1 else CAMINHO_PADRAO

    # Carrega e valida a base antes de gerar as tabelas.
    base = carregar_base(caminho_base)
    validar_colunas(base, [VARIAVEL_DISCRETA, VARIAVEL_CONTINUA])

    # Cria a tabela e os insights da variavel quantitativa discreta.
    tabela_discreta, dados_discretos = criar_tabela_discreta(base, VARIAVEL_DISCRETA)
    estatistica_discreta = criar_tabela_estatistica_descritiva(dados_discretos)
    insights_discretos = gerar_insights_discreta(
        tabela_discreta,
        dados_discretos,
        VARIAVEL_DISCRETA,
    )

    # Cria a tabela e os insights da variavel quantitativa continua.
    tabela_continua, dados_continuos = criar_tabela_continua(base, VARIAVEL_CONTINUA)
    estatistica_continua = criar_tabela_estatistica_descritiva(dados_continuos)
    insights_continuos = gerar_insights_continua(
        tabela_continua,
        dados_continuos,
        VARIAVEL_CONTINUA,
    )

    dados_anos_cost = preparar_dados_anos_cost(base)
    tabela_tendencia_ano = criar_tabela_tendencia_ano(dados_anos_cost)
    insights_tendencia_ano = gerar_insights_tendencia_ano(dados_anos_cost)

    # Mostra os resultados finais no terminal.
    imprimir_resultado(
        f"Tabela de frequencia - variavel discreta: {VARIAVEL_DISCRETA}",
        tabela_discreta,
        insights_discretos,
    )
    imprimir_tabela(
        f"Estatistica descritiva - variavel discreta: {VARIAVEL_DISCRETA}",
        estatistica_discreta,
    )
    imprimir_resultado(
        f"Tabela de frequencia - variavel continua: {VARIAVEL_CONTINUA}",
        tabela_continua,
        insights_continuos,
    )
    imprimir_tabela(
        f"Estatistica descritiva - variavel continua: {VARIAVEL_CONTINUA}",
        estatistica_continua,
    )
    imprimir_tabela(
        f"Tendencia de preco por ano: {VARIAVEL_CONTINUA}",
        tabela_tendencia_ano,
    )
    plotar_tendencia_ano(tabela_tendencia_ano)
    imprimir_insights(
        "Insights sobre a tendencia de precos ao longo dos anos",
        insights_tendencia_ano,
    )


if __name__ == "__main__":
    main()
