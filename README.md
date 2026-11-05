# Projeto Sismos

Indicador de potencial de estrago sísmico por país, cruzando eventos do **USGS**, mortes do **NOAA**
e PIB per capita / densidade populacional do **Banco Mundial**.

| Arquivo | O que faz |
|---|---|
| `wrangling_sismos.py` | Limpa e cruza as bases brutas (`dados/`) e gera os CSVs prontos para análise (`saida/`). |
| `Analise_Sismos.ipynb` | Notebook completo: **roda o wrangling**, faz a exploração e a clusterização dos países (K-Means). |
| `checagens.py` | Conferência rápida dos CSVs de saída. |

## Estrutura

```
projeto_sismos/
├── Analise_Sismos.ipynb
├── wrangling_sismos.py
├── checagens.py
├── requirements.txt
├── dados/                       # bases brutas (entrada)
│   ├── usgs_1990_2023_m55.csv
│   ├── noaa_earthquakes.tsv
│   ├── API_NY.GDP.PCAP.CD_...csv            # PIB per capita (Banco Mundial)
│   ├── API_EN.POP.DNST_...csv               # densidade populacional (Banco Mundial)
│   ├── Metadata_Country_...csv              # metadados dos países (Banco Mundial)
│   ├── ne_50m_admin_0_countries.zip         # shapefile Natural Earth (países)
│   └── covariaveis_extra.csv                # PIB/densidade de Taiwan (ausente no BM)
└── saida/                       # CSVs gerados pelo script
```

## 1. Preparar o ambiente

Requer **Python 3.10 ou superior** (testado com 3.12).

```bash
# dentro da pasta projeto_sismos/
python -m venv .venv

source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows (PowerShell / cmd)

pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows:** se a instalação do `geopandas` falhar, atualize o pip e tente de novo. As versões atuais
> já vêm com wheels prontos, então não precisa instalar GDAL separadamente.

## 2. Ajuste de caminhos (para rodar tudo da raiz do projeto)

O `wrangling_sismos.py` procura os arquivos de entrada pelo nome simples (ex.: `usgs_1990_2023_m55.csv`)
e grava em `./saida`, tudo relativo à pasta **de onde o comando é executado**. Como os dados ficam em
`dados/`, rodar da raiz dá `FileNotFoundError`, e o **notebook**, que roda a partir da raiz, tem o mesmo problema.

A solução mais simples é apontar o `CFG` para `dados/`. No `wrangling_sismos.py`, logo depois do bloco
`CFG = dict(...)` e antes de `OUT = Path("saida")`, acrescente:

```python
DATA = Path("dados")
for _k in ("usgs", "noaa", "wb_gdp", "wb_dens", "wb_meta", "countries_shp", "wb_extra"):
    CFG[_k] = str(DATA / CFG[_k])
```

Com isso, script, notebook e `checagens.py` funcionam todos a partir da raiz e a saída vai direto para `saida/`.
(Testei esse ajuste rodando o script da raiz: funciona.)

> Se preferir **não** alterar o código, o script (só ele, não o notebook) roda de dentro de `dados/`:
> `cd dados && python ../wrangling_sismos.py`. A saída vai para `dados/saida/`; mova para a raiz com
> `Move-Item dados\saida saida` (PowerShell) ou `mv dados/saida saida` (Linux/macOS).

## 3. Rodar o script (`wrangling_sismos.py`)

Da raiz do projeto, com o ambiente virtual ativado:

```bash
python wrangling_sismos.py
```

Ele imprime o progresso (match USGS ↔ NOAA, declustering, casos conhecidos como Sumatra 2004, Haiti 2010,
Tohoku 2011) e termina com um `[RESUMO]`. Resultado esperado:

```
[RESUMO] events.csv: 13315 | countries.csv: 218
  events_modelo.csv: 3407 mainshocks M>=6.0 | Alta_Mortalidade=1: 92 (2.7%) | log_mortes==0: 88%
  countries_cluster.csv: 93 países
```

Arquivos gerados em `saida/`:

| Arquivo | Conteúdo |
|---|---|
| `events_modelo.csv` | Mainshocks M≥6 com covariáveis, pronto para regressão/classificação |
| `countries_cluster.csv` | Países com M6+ e covariáveis, pronto para clustering |
| `events.csv` | Todos os eventos USGS (com flag `eh_mainshock`) |
| `countries.csv` | Todos os países do Banco Mundial (inclusive sem sismos) |
| `noaa_sem_match.csv` | Eventos NOAA com mortes que não casaram com o USGS (para inspeção) |
| `events_sem_covariaveis.csv` | Eventos sem PIB/densidade |

Para conferir a saída: `python checagens.py` (da raiz).

Os parâmetros (anos, magnitude mínima, tolerâncias do match, janela de declustering etc.) ficam no
dicionário `CFG` no topo do script.

## 4. Rodar o notebook (`Analise_Sismos.ipynb`)

O notebook **não lê mais só os CSVs prontos**: a primeira parte executa o wrangling chamando
`wrangling_sismos.main()`, então ele depende do script e dos dados de `dados/`.

**Pré-requisitos**

1. Ambiente instalado (passo 1) e ajuste de caminhos feito (passo 2).
2. O notebook precisa estar na **mesma pasta** do `wrangling_sismos.py` (a raiz do projeto), porque faz
   `import wrangling_sismos`.
3. A função `main()` do script precisa **devolver os quatro objetos** que o notebook espera. Na célula
   do wrangling o notebook faz:

   ```python
   ev, paises_completos, eventos, paises = wrangling_sismos.main()
   ```

   Ou seja, o final do `main()` precisa ter:

   ```python
   return ev, paises, mod, clu   # events, countries, events_modelo, countries_cluster
   ```

   (na versão do script que veio no zip o `main()` não tem `return`; sem ele a célula falha com
   `TypeError: cannot unpack non-iterable NoneType object`.)

**Como executar**

- **Jupyter local:** com o ambiente ativado, na raiz do projeto:

  ```bash
  jupyter lab Analise_Sismos.ipynb
  ```

  Depois *Run → Run All Cells*.
- **VS Code:** instale as extensões *Python* e *Jupyter*, abra o `.ipynb`, escolha como kernel o `.venv`
  (canto superior direito) e clique em *Run All*.

**O que acontece, célula a célula**

| Etapa | O que faz |
|---|---|
| `%pip install ...` | Instala as bibliotecas no kernel. Pode pular se você já fez `pip install -r requirements.txt`. |
| Teste de ambiente | Importa numpy, pandas, geopandas e sklearn e imprime `Ambiente configurado!`. |
| Wrangling | Roda o script (leva alguns minutos, principalmente a atribuição de países aos epicentros) e imprime `Eventos: (3407, 36)` e `Países: (93, 19)`. |
| Exploração | Estatísticas descritivas e gráfico magnitude × log(1 + mortes). |
| Clusterização | K-Means com K = 5, PCA, validação e geração de `paises_clusters.csv`. Resultado de referência: silhouette = 0,305. |

**Observações**

- A célula de clusterização ainda lê `countries_cluster.csv` **do GitHub** (`raw.githubusercontent.com/EnzoSottoRib/projeto_sismos/...`),
  então precisa de internet e usa o arquivo publicado no repositório, não o que você acabou de gerar. Para usar o local,
  troque as duas linhas de leitura por `paises = pd.read_csv("saida/countries_cluster.csv")`
  (ou simplesmente remova a leitura, já que `paises` vem do `main()`).
- O notebook grava `paises_clusters.csv` na pasta onde roda (a raiz).
- **Google Colab:** o botão antigo do notebook não funciona mais sozinho, porque agora ele precisa do script e dos
  arquivos de `dados/`. Para usar no Colab seria preciso clonar o repositório antes; este caminho não foi testado.

## Problemas comuns

| Erro | Causa / solução |
|---|---|
| `FileNotFoundError: 'usgs_1990_2023_m55.csv'` | Caminhos relativos sem o ajuste do passo 2. Aplique o ajuste ou rode o script de dentro de `dados/`. |
| `FileNotFoundError: 'saida/noaa_sem_match.csv'` (no `checagens.py`) | A pasta `saida/` não está na raiz. Rode o script (passo 3) ou mova `dados/saida` para a raiz. |
| `TypeError: cannot unpack non-iterable NoneType object` (notebook) | O `main()` do script não devolve os 4 objetos. Veja o item 3 dos pré-requisitos do notebook. |
| `ModuleNotFoundError: sklearn` / `geopandas` / etc. | Ambiente virtual não ativado, ou o kernel do notebook não é o `.venv`. Rode `pip install -r requirements.txt` e selecione o kernel certo. |
| `ModuleNotFoundError: wrangling_sismos` | O notebook não está na mesma pasta do script (ou o kernel foi aberto em outra pasta). |
| `URLError` / `HTTP Error` na clusterização | Sem internet ou caminho do GitHub mudou. Use a leitura local descrita nas observações. |
| `[BM] modelo p/ Taiwan criado: covariaveis_extra_MODELO.csv` | Aviso normal quando o `covariaveis_extra.csv` não é encontrado na pasta de dados. |
