# 📂 Pasta `scripts` — Predição de Gênero + Atualização no SQL Server

Este documento descreve o pipeline **rondon florest** para **inferência de gênero** a partir de dados da CAPES/Sucupira, com **atualização direta** na tabela alvo do SQL Server.

> **Resumo do fluxo:**  
> 1) Treina **RF contextual** (features institucionais/produção)  
> 2) Treina **modelo por nome** (TF-IDF de caracteres + LR calibrada)  
> 3) Aplica regras **Bayes (nome, UF)** e **Bayes (nome)**  
> 4) **Adapta limiares** até atingir a meta de cobertura  
> 5) Aplica **fallback** para garantir 100%  
> 6) Grava **colunas de saída** na tabela e **registra LOG**  

---

## 📦 Entrada, Saída e Pré-requisitos

**Entrada (SQL Server)**  
- Banco: `IC`  
- Tabela: `dbo.STEM_Y`  
- Chave: `ROW_ID`  
- Nome do autor: `NM_AUTOR`  
- Coluna alvo: `GENERO` (`FEMININO`, `MASCULINO`, `INDETERMINADO`)  

**Features exigidas**  
`NM_AREA_BASICA`, `NM_MODALIDADE_PROGRAMA`, `NM_GRAU_PROGRAMA`, `DS_SITUACAO_PROGRAMA`,  
`NM_REGIAO`, `SG_UF_PROGRAMA`, `DS_DEPENDENCIA_ADMINISTRATIVA`, `NM_ENTIDADE_ENSINO`,  
`NM_PROGRAMA_FOMENTO`, `NM_FINANCIADOR`, `NM_NATUREZA_FINANCIAMENTO`,  
`NM_TIPO_PRODUCAO`, `NM_SUBTIPO_PRODUCAO`, `TP_AUTOR`

**Saídas gravadas**  
- `GENERO_PRED`, `PROB_GENERO_PRED`, `FOI_IMPUTADO`, `MODO_IMPUTACAO`, `CONF_FONTE`, `IMPUTACAO_FORCADA`

**Artefatos gerados**  
- `modelo_genero.joblib` (RF contextual)  
- `modelo_genero_nome.joblib` (TF-IDF + LR calibrada)  
- `modelo_genero_metrics.json` (métricas da RF)  
- `modelo_genero_nome_metrics.json` (métricas do modelo de nome)  
- `checkpoint_predicao.log` (log textual)  
- Tabela de log no banco: `dbo.LOG_IMPUTACAO_GENERO`

---

## 🔁 Pipeline — Visão Geral

```mermaid
flowchart TD
  A["Dados SQL (STEM_Y)"] --> B["Pré-processamento"];
  B --> C["Treino RF (contexto)"];
  B --> D["Treino Nome (TF-IDF + LR calibrada)"];
  B --> E["Estatísticas Bayes (nome, UF) e (nome)"];
  F["Casos INDETERMINADO"] --> G["Predição em cascata"];
  C --> G;
  D --> G;
  E --> G;
  G --> H["Adaptação de limiares até meta"];
  H --> I["Fallback para 100%"];
  I --> J["Staging + UPDATE em STEM_Y"];
  J --> K["LOG_IMPUTACAO_GENERO"];
