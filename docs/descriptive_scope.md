# Protocol V2 × cost addendum — prospective amendment to the scope of claims

**Decision:** ratified on 2026-11-14 by the responsible researcher. **Nature:** prospective scientific amendment to the scope of the conclusions, recorded after the MLP development pilot and before the campaign. It is not an execution authorization. The five-criteria proposal it originated from is not part of this folder; the approved text is reproduced below.

## Ratified text

The approved text is quoted in its original language (Portuguese), because that is the wording that was ratified. An English translation follows.

> **Aprovo uma emenda prospectiva ao alcance das alegações do projeto**, preservando os 160 planos, os dois datasets, os dois modelos, as oito estratégias e as cinco seeds.
>
> 1. A conclusão principal será uma comparação descritiva nos quatro estratos dataset × modelo. PH global × FedAvg contínuo permanece como contraste principal.
> 2. Preservo o Macro-F1 global de classes fixas por janela, a média temporal não ponderada, o pareamento e a publicação obrigatória dos erros por classe, suportes e custos.
> 3. Na LR determinística, seeds diferentes não serão tratadas como replicações independentes. No MLP, os cinco contrastes e sua dispersão serão relatados condicionados às coortes e aos horizontes fixos.
> 4. As margens `−0,02` e `0,03` permanecerão referências históricas provisórias, sem uso para declarar não inferioridade ou segurança preservada. Custos serão comparados descritivamente, sem alegação de H2 inferencial.
> 5. Não será declarado sucesso na regra inferencial H1/H2 original. Qualquer inferência futura dependerá de emenda e método próprios, sem constituir condição indefinida para encerrar este estudo.
> 6. Pares essenciais incompletos serão identificados como não avaliáveis, sem imputação. Ausência de adaptações e resultados desfavoráveis serão preservados. Sem verdade independente de mudança, `detection_delay` será não identificável.
>
> A emenda não autoriza a campanha, não dispensa Gotham nem a etapa de análise confirmatória e não altera retrospectivamente os resultados development.

**English translation.**

> **I approve a prospective amendment to the scope of the project's claims**, preserving the 160 plans, the two datasets, the two models, the eight strategies and the five seeds.
>
> 1. The main conclusion will be a descriptive comparison in the four dataset × model strata. Global PH × continuous FedAvg remains the main contrast.
> 2. I preserve the per-window global Macro-F1 with fixed classes, the unweighted temporal mean, the pairing, and the mandatory publication of per-class errors, supports and costs.
> 3. For the deterministic LR, different seeds will not be treated as independent replications. For the MLP, the five contrasts and their dispersion will be reported conditional on the fixed cohorts and horizons.
> 4. The `−0.02` and `0.03` margins will remain provisional historical references, not used to declare non-inferiority or preserved safety. Costs will be compared descriptively, with no inferential H2 claim.
> 5. Success under the original inferential H1/H2 rule will not be declared. Any future inference will depend on its own amendment and method, and will not be an open-ended condition for closing this study.
> 6. Incomplete essential pairs will be identified as not evaluable, without imputation. Absence of adaptations and unfavorable results will be preserved. Without an independent ground truth of change, `detection_delay` will be not identifiable.
>
> The amendment does not authorize the campaign, does not waive Gotham or the confirmatory analysis stage, and does not retroactively change the development results.

## Prospective application and precedence

This decision replaces, **only for the future interpretation of the campaign**, the earlier obligation to declare inferential H1 non-inferiority in the four strata and the associated inferential H2 success. The historical fields of protocol V2 ([`protocol/protocol_v2.yaml`](protocol/protocol_v2.yaml)) (margin `−0.02`, guardrails `0.03`, 95% CI, bootstrap and the H1/H2/H3/H5 success rule) remain intact and identifiable as the earlier frozen contract; they are neither executed nor silently reinterpreted as a validated confirmatory test. Any operational specification that still emits a non-inferiority, preserved-safety or H1/H2-success verdict from those fields must stay inactive until it is explicitly aligned with this amendment. The 160-plan campaign matrix does not change.

The main report must show the four strata separately, comparable pairs by seed and horizon, and the per-window global Macro-F1 with fixed classes and its unweighted temporal mean. It must also publish TP/FP/TN/FN, recalls/FNR/FPR with support, per-client performance and the temporal tail when those aggregates are available, alarms/adaptations and costs per phase. Undefined values are never converted to zero. For the deterministic LR, repeated seed IDs do not increase the number of independent realizations; for the MLP, the five contrasts are observations conditional on the fixed cohorts and horizons, with no automatic CI. Update costs and total cost are reported separately; logical bytes do not mean physical traffic.

A missing essential pair makes **that contrast** not evaluable, with its cause and coverage published, without imputation or silent exclusion. This does not allow replacing one stratum with another or declaring joint success. `detection_delay` is `NOT_IDENTIFIABLE` when there is no independent ground truth of change; PH alarms are not themselves the drift ground truth. Unfavorable results, zero alarms and zero adaptations are results to be preserved.

## What the amendment does not change

The amendment grants no data access, does not authorize the campaign and does not admit datasets beyond those defined in the protocol. At the date of the decision, none of the 160 plans had been executed.
