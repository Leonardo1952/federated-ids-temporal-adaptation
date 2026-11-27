# Normative authority of the TON_IoT binary mapping in the window metrics (development)

**Status:** normative rule ratified by the responsible researcher on 2026-11-05, for the internal development-pilot route. It is not an execution authorization. In the code of this folder the rule is exercised only through a synthetic test authority (`ton_label_mapping_authority.py`), which checks the SHA-256 of this document.

## Ratified text

The approved text is quoted in its original language (Portuguese); internal stage names were replaced by the functional names used in this folder. An English translation follows.

> Como pesquisador responsável e custodiante, ratifico prospectivamente uma autoridade de mapeamento das métricas por janela distinta da autoridade de preparação da janela 0, exclusivamente para o piloto exploratório TON_IoT de desenvolvimento. Ela implementa a regra científica window-metrics-delta-01 já ratificada: target=0 benigno, target=1 ataque; Unknown permanece fora de TP/FP/TN/FN e é contado separadamente; qualquer valor não previsto falha fechado. A versão do mapeamento será window-metrics-ton-development-mapping-v1, com documento canônico e SHA-256 verificável.
>
> Cada emissão operacional deverá vincular o release/linhagem admitidos, recibo de integridade, 45 shards development e seus hashes, run/plan, seed, estratégia, versão de código, configuração, manifesto de entrada, GO específico e destino. A âncora de mapeamento será comprometida antes de cada avaliação; o vínculo mapeamento↔predição↔avaliação e a projeção das métricas por janela serão persistidos no checkpoint da janela, com consumo e auditoria verificáveis. Somente evidências já comprometidas poderão ser recuperadas após reinício.
>
> O escopo de campos é target e supervision_status no canal de avaliação do vínculo físico e do contrato de leitura das bases; esta autoridade não concede acesso adicional a labels de preparação, janela 1, features não autorizadas ou confirmation. Esta decisão normativa não é GO, não autoriza leitura ou treinamento real, não altera métricas por janela v1, pacote da execução, o estimando, as estratégias ou a matriz da campanha, e não se estende a Gotham, atestação de entrada ou execução confirmatória. O primeiro uso real dependerá de implementação fechada validada, compatibilidade dos checkpoints, demais gates e autorização operacional específica.

**English translation.**

> As responsible researcher and custodian, I prospectively ratify a window-metrics mapping authority distinct from the window-0 preparation authority, exclusively for the exploratory TON_IoT development pilot. It implements the already ratified scientific rule window-metrics-delta-01: target=0 benign, target=1 attack; Unknown stays outside TP/FP/TN/FN and is counted separately; any unexpected value fails closed. The mapping version will be window-metrics-ton-development-mapping-v1, with a canonical document and a verifiable SHA-256.
>
> Each operational issuance must bind the admitted release/lineage, the integrity receipt, the 45 development shards and their hashes, run/plan, seed, strategy, code version, configuration, input manifest, specific GO and destination. The mapping anchor will be committed before each evaluation; the mapping↔prediction↔evaluation binding and the window-metrics projection will be persisted in the window checkpoint, with verifiable consumption and audit. Only evidence already committed may be recovered after a restart.
>
> The field scope is target and supervision_status in the evaluation channel of the physical binding and the dataset-reading contract; this authority grants no additional access to preparation labels, window 1, unauthorized features or confirmation. This normative decision is not a GO, does not authorize real reading or training, does not change window metrics v1, the run bundle, the estimand, the strategies or the campaign matrix, and does not extend to Gotham, the entry attestation or confirmatory execution. The first real use will depend on a closed, validated implementation, checkpoint compatibility, the remaining gates and a specific operational authorization.

## Scope of this record

The decision removes the pending ratification of the normative window-metrics mapping rule described above. It does not automatically promote the artificial synthetic-fixture-authority, does not make the real route executable and does not approve an operational implementation not yet reviewed.

The window-0 preparation authority remains a separate decision. The physical-binding and dataset-reading contracts and the specific operational authorization are still required.

UTF-8 document, LF line endings, no BOM and exactly one terminal newline. The SHA-256 is computed over these bytes. This file contains no active authorization and no inventory of real data.
