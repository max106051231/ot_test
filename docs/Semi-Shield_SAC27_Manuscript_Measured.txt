Semi-Shield: Obligation-Closed Evidence and Approval-Bound
Release for OT Cybersecurity Assessments
ABSTRACT
Operational technology (OT) assessments must justify a bounded
requirement using authorized records for the correct site, assets,
interval and policy version. A cited login or an absence of alerts
does not establish that observation was complete. We specify
Semi-Shield and its core method, the Coverage-aware Obligation
and Read-set Gate (CORG). A hand-authored obligation contract
defines sufficient positive and violation witnesses, authoritative
source selectors, and coverage conditions. Retrieval and a local
model propose atomic claims; CORG independently closes the re-
quired evidence queries, checks raw spans and scope, and assigns
SUPPORTED, VIOLATED or INSUFFICIENT. The proof certificate
records both cited witnesses and query dependencies, including
empty results and interval-scoped finality/coverage claims. Hu-
man approval binds the exact report and certificate; the publisher
re-evaluates those dependencies in a serializable local transaction
before release. This targets late, uncited counterevidence that
a cited-bundle digest alone can miss. We specify a paired OT
benchmark, public auxiliary probes, strong rule and action-bound
approval baselines, and metrics for error, utility and release safety.
We report implementation traces and measured judgment and release outcomes on a 522-packet held-out benchmark (Table 2–6). The
target is scoped OT assurance support, not certification or control
of industrial equipment.
KEYWORDS
OT cybersecurity, evidence obligations, three-valued assessment,
read-set validation, approval-bound publication, retrieval aug-
mented generation
Measured-results revision (Semi-Shield ot_test). Scoring scripts and case manifest: data/eval_sac/. Table cells report observed values; synthetic gold unless noted.
1 INTRODUCTION
1.1 Motivation from OT evidence practice
Event monitoring and audit assessment answer different questions.
Monitoring systems describe observed activity, while a control
assessment must establish which requirement applies, which
assets and interval are covered, and whether available records
support the proposed judgment. A switch login record can iden-
tify an observed management session. It does not independently
establish the legitimacy of that session, complete monitoring
coverage or organization-wide access-control effectiveness. Poli-
cies, authorized-account records, maintenance approvals, network
paths and collector coverage may be decisive.
Semiconductor facilities may require sensitive operational
records to remain on premise. This deployment constraint mo-
tivates local inference and controlled evidence retrieval, rather
SAC ’27, April 5–9, 2027, Gwangju, South Korea
2027.
than establishing that every cloud GRC product is unsuitable. OT
performance, reliability and safety constraints also require explicit
boundaries around automation [ 2]. Semi-Shield assists analysis
and report preparation within those boundaries. It issues no PLC
commands, configuration changes or production blocking actions.
1.2 Research gap and intended contributions
Existing RAG systems, evidence-linked argument graphs, ground-
ing checkers and action-bound approvals address important parts
of this workflow (Section 2). The unresolved question for this
study is whether an OT-specific obligation can identify the com-
plete decision-relevant read set , including observations needed to
justify absence, and keep that assessment valid until the exact
report is released. A cited identifier can resolve correctly while its
record has the wrong scope; even an unchanged cited bundle can
become misleading when an uncited counterexample arrives after
approval.
The proposed contributions are (i) a machine-checkable OT
obligation contract with explicit positive, negative and coverage
requirements; (ii) CORG, which certifies a three-valued judgment
and revalidates obligation-derived evidence dependencies at pub-
lication; and (iii) an evaluation that separates semantic judgment
from release enforcement against public and matched strong
baselines. Action digests, rule engines and approval gates are
established techniques; the intended increment is the construction
and testing of OT-specific evidence closure. These contributions
require implementation and empirical validation. Graph retrieval
and fine-tuning are supporting factors, not presumed advances.
1.3 Research questions
RQ1 asks whether obligation closure reduces unsupported af-
firmative judgments beyond a generic verifier at comparable
answerable-case resolution. RQ2 asks whether binding approval
to the obligation-derived read set prevents release after relevant
post-approval changes while avoiding re-review on irrelevant
changes, relative to strong action-bound gates. RQ3 asks whether
graph traversal improves eligible evidence retrieval under iden-
tical filters and budget. Model adaptation is optional and is not
needed to establish RQ1–RQ2.
2 BACKGROUND AND RELATED WORK
2.1 Control context and operational proof
obligations
ISO/IEC 27001 addresses an information security management sys-
tem, including organizational processes and risk management [ 1].
Its control references do not turn a single event into a complete cer-
tification judgment. We therefore evaluate organization-defined,
scoped proof obligations that can be connected to control refer-
ences [3]. Clauses concerning organizational context or leadership
SAC ’27, April 5–9, 2027, Gwangju, South Korea
require appropriate organizational records; OT event counts can-
not substitute for them.
We organize bounded tasks into six OT monitoring families (Ta-
ble 1); these are collection and evaluation strata, not six automatic
control pass/fail tests. Applicability and broader control effective-
ness remain the responsibility of the organization and its autho-
rized reviewers. NIST SP 800-53A provides assessment objectives
and methods, and OSCAL Assessment Results can encode subjects,
observations, findings and their time-bounded provenance; neither
supplies a gold judgment for this OT task [ 11, 12].
2.2 Compliance retrieval and evidence
reasoning
Bolton et al. study multi-stage RAG for railway OT cybersecurity
questions [13]. ReCon benchmarks resource-constrained compar-
ison of policy documents to ISO 27002, but explicitly measures
documentary coverage rather than operational implementation;
its masked data are available only on request [ 14]. Industrial
assurance-case RAG combines claim–argument–evidence (CAE)
structures with NLI [ 15]. Compliance-by-construction argument
graphs already join typed claims, RAG drafts, admissibility and
coverage checks, provenance and human oversight [ 16]. Con-
sequently, graph-linked evidence or an NLI check alone is not
the proposed contribution here. CORG addresses the narrower
question of whether bounded OT observations, including their
collection completeness, discharge an applicable operational
obligation.
GraphRAG local search combines graph relationships and
source text [4]; the proposed graph-assisted retriever is only a can-
didate generator, never the authority defining a complete evidence
universe. MiniCheck provides an independent grounding checker
[5], while RAGTruth measures generic RAG hallucinations [ 6].
FEVEROUS and ContractNLI evaluate multi-piece evidence and
three-way inference, and ACE evaluates interacting contract
clauses [ 17, 18, 19]. Their third labels respectively mean insuffi-
cient encyclopedic evidence, unmentioned contract content and
non-applicability; none directly encodes OT collector coverage or
approval validity.
2.3 Approval and publication boundaries
OPA documents an external policy decision point consulted by
an agent’s execution harness [ 20]; Microsoft Agent Governance
Toolkit (AGT) implements action-bound human approval with ex-
act action digest, policy/chain versions, expiry and execution-time
validation [21]. Loopjacking reproduces approval of one action fol-
lowed by execution of another, and SOUNDGATE studies effect
mediation during pause, replay and cancellation [ 22, 23]. Typed
provenance and assertion-time mediation have also been proposed
for persistent agents [24]. These works establish that fail-closed ac-
tion approval and hash binding are prior techniques. They do not,
by themselves, derive the site-, time- and source-specific evidence
closure of an OT obligation, including uncited counterevidence and
justified empty query results. AGT can enforce such a dependency
digest if supplied one; Section 5 tests whether constructing that
digest from the obligation improves selective release validity, not
whether AGT’s approval primitive is defective.
3 TASK AND THREAT MODEL
3.1 Assessment input and output
An assessment request𝑞0 = (𝑟, 𝑢, 𝑆,𝑊 , 𝑣𝑝, 𝑣𝑡 ) fixes an organization-
defined requirement 𝑟 , authorized principal 𝑢, site/assets/zones
𝑆, event interval 𝑊 , and the governing policy/topology versions
𝑣𝑝, 𝑣𝑡 determined at assessment. The registry’s ingest cutoff is a
separate snapshot parameter: 𝜏𝑎 for assessment and 𝜏𝑟 ≥ 𝜏𝑎 for re-
lease. At release, current applicability and versions are rechecked;
a changed governing version invalidates the approval rather than
silently changing 𝑞0. Applicability is established from authorita-
tive policy before the three-way task: a non-applicable control is
out of scope, not INSUFFICIENT. The output consists of a decision
𝑦 ∈ { SUPPORTED, VIOLATED, INSUFFICIENT}, atomic claims,
source spans, contradictions and gaps. SUPPORTED is limited to
𝑞0 at its snapshot; VIOLATED needs an eligible counterexample
and any absence premise it depends on; INSUFFICIENT covers
unresolved gaps or conflicting witnesses. Invalid or blocked
model output is a distinct processing failure, never a correct
INSUFFICIENT by default.
Publication state is separate: draft, review required, human
approved, rejected or released. Both violations and insufficient-
evidence findings may be included in a reviewed report with their
qualifications. No assessment label independently grants permis-
sion to publish, and no AI reviewer can issue a human approval
token.
3.2 Adversary and trust boundaries
The adversary may place instructions in retrieved tickets, docu-
ments, syslog fields or graph descriptions; substitute a similar arti-
fact from another asset or site; exploit stale policies and topology;
invent citations; or trigger fallback and guard-unavailable paths.
After approval, ingestion may add an uncited, relevant counterex-
ample, change a coverage record, or replace an approved object.
For a positive absence claim already certified as final, such a late
record is a correction or breach of the source-finality contract, not
routine on-time ingestion. Trusted components are authenticated
acquisition, a transactionally versioned local authority view, ac-
cess enforcement, the publication transaction and human approval
identity. Trust in a collector is an assumption unless source authen-
tication is implemented and tested.
A local hash chain cannot establish that a source event actually
occurred. A writer able to replace an entire registry can recom-
pute a self-consistent chain; detection needs an independently pro-
tected checkpoint or equivalent control. Complete source selectors
and coverage attestations are required assumptions for absence-
based inference. Model compromise, compromised trusted review-
ers, physical process safety and synchronous atomicity with exter-
nal source systems are outside scope. The release invariant is con-
ditional on what has been ingested into the local authority view,
not on unsynchronized external truth; these exclusions constrain
any measured safety claim.
Semi-Shield: Obligation-Closed OT Evidence SAC ’27, April 5–9, 2027, Gwangju, South Korea
Table 1: Monitoring categories and evidence requirements for scoped tasks.
Monitoring category Candidate context Needed beyond an event
Management access A.5.15 Account authority, MFA/session correlation, approved path and collection cover-
age
Supplier or asset context A.5.19 only with supplier records Supplier identity, contractual requirements and approved maintenance access
Malware protection A.8.7 Protection deployment, coverage, policy and response records
Vulnerability or patch handling A.8.8 Affected inventory, advisory, installed version, approval or documented exception
Configuration changes A.8.9 and A.8.32 Configuration versions, authorized change and maintenance window
Protected management traffic A.8.24 Applicable cryptographic policy, actual protocol/configuration and session evi-
dence
4 PROPOSED METHOD
4.1 Architecture and role permissions
Figure 1 separates collection, drafting, verification and release
authority. Collector registers evidence and preserves source lo-
cations. Auditor receives a bounded evidence bundle and creates
a draft. Reviewer performs checks and assists human triage. Re-
porter accepts only publication-gate outputs. A deterministic
orchestrator may implement these roles; role names alone do not
imply autonomous multi-agent planning.
Semi-Shield assessment and release boundary
Collector − →versioned local authority view − →eligible retrieval
− →Auditor draft
?? y ?? y
Source finality and coverage − →CORG obligation closure and
proof certificate
?? y
Independent claim checks − →human approval − →serializable
release gate − →persisted report
Every revision, fallback and rejected gate returns to review.
Figure 1: Proposed integrated architecture. Retrieved data
is checked before generation, every revision returns to ver-
ification, and publication requires an unchanged approved
snapshot. Normative records and observed evidence remain
distinct.
4.2 Source preserving evidence registration
Each evidence object records evidence_id, source type, raw-byte
digest, source locator and immutable span offsets, site and asset,
event time and ingest time with timezone, collector identity, parser
version, simulation flag and access policy. Canonical metadata is
digested separately from raw bytes. Summaries and parsed events
retain parent evidence identifiers so that derived content never re-
places its raw source. The registry records a sequence number and
predecessor digest; verification checks both the referenced payload
and relevant chain segment against a trusted checkpoint when one
exists [10].
Collector heartbeats, loss indicators, retention interval, covered
assets and event-time finality attestations are first-class evidence.
A source contract states a maximum accepted lateness 𝐿𝜎 and a
protected watermark 𝐹𝜎 : events with event time at or before 𝐹𝜎
have either been ingested or must arrive as an explicit correction or
retraction. An absence premise for 𝑊 requires 𝐹𝜎 ≥ max(𝑊 ), no
intersecting loss or retention gap, and authoritative source cover-
age. Before that point the premise is unresolved, even if no event is
currently visible. A correction after finality invalidates the scoped
coverage claim and starts review or revocation for affected reports;
the gate can prevent only a release occurring after that correction
is ingested. Hash validity, source authenticity, access entitlement,
scope eligibility and semantic support are distinct properties and
are logged independently. Event time is distinct from ingest time.
4.3 Core method: coverage-aware obligation
and read-set gate
CORG compiles each applicable, human-authored obligation 𝑂𝑟 =
(𝐴𝑟 , Σ𝑟 , 𝑃+𝑟 , 𝑃 −𝑟 , 𝐶𝑟 , 𝐸𝑟 ) into an AND/OR proof graph. 𝐴𝑟 is an ap-
plicability and version rule; typed selectors Σ𝑟 enumerate author-
itative records; 𝑃 +𝑟 and 𝑃 −𝑟 specify sufficient positive and decisive
violation branches; 𝐶𝑟 supplies coverage/finality premises; and 𝐸𝑟
records exceptions and source precedence. Each selector declares
an authority, source schema, site/asset join keys, event-time inter-
val predicate, ingest-cutoff parameter, entitlement rule and record
validity interval. Alternative evidence combinations are explicit
OR branches. A selector is a deterministic closure query: it enumer-
ates every authoritative record relevant to the bounded predicate,
including an empty result. RAG may rank candidate spans but can-
not define the counterexample population.
The compiler type-checks join keys, time bases, authority
and version references; expands each proof branch into required
factual atoms; and requires a closure selector for every atom
that could overturn that branch. Every negative or universal
atom must point to an interval-specific coverage and finality
predicate. An unbound selector, uncovered branch, ambiguous
source precedence or non-final absence premise fails compilation
or leaves that branch unresolved. These checks prove internal
contract consistency, not that an expert chose every real-world
source. Two OT reviewers therefore compare each template with
an independent source inventory and assessment rubric, execute
positive, negative, missing-source and late-correction mutation
tests, and report selector recall against independently enumerated
gold records. Templates, compiler version and review decisions
are frozen before the held-out evaluation.
For a remote-maintenance obligation, Σ𝑟 selects management
sessions, the authoritative approval register, MFA correlation,
SAC ’27, April 5–9, 2027, Gwangju, South Korea
jump-host routes and collector coverage for the requested assets
and window. One positive branch requires every eligible session
to match an authorized account, approved window/path and
required MFA, plus finality for the session and approval sources
over 𝑊 . One violation branch accepts an authenticated session
paired with a decisive, current deny record; another requires
proof that no matching approval exists, which itself needs a final
authoritative approval-register lookup. A direct violation need
not wait for unrelated collectors. A policy document is normative
context and cannot substitute for operational witnesses.
For each selector 𝜎 ∈ Σ𝑟 , CORG stores a sorted digest of its
scoped result set , including the zero-result case, and the selector
definition. The query fixes event interval 𝑊 but uses the current
ingest cutoff 𝜏. Coverage is represented by a normalized interval-
scoped claim: finality through 𝑊 , intersecting loss or retention
gaps and any correction or retraction affecting 𝑊 . A global data-
base revision, later finality watermark or heartbeat is logged for
audit but excluded from approval equality when it cannot change
this scoped claim. A newly ingested relevant record or retroactive
loss intersecting 𝑊 changes the selector result or claim. Where the
source cannot attest a stable complete window, absence-based pos-
itive judgments remain INSUFFICIENT. CORG validates raw bytes,
cited offsets, span-level site, asset and time, authorization, simu-
lation state, applicable policy and topology versions and source
authority. Verified relationships must resolve to raw records. This
produces a predicate read set 𝑅(𝑞0; 𝜏): cited witnesses plus bounded
query results and normalized coverage facts on which the verdict
depends. It is narrower than the whole corpus and broader than
cited documents.
For example, a Fab-A affirmative report cites two SW-07 main-
tenance sessions after the source attests finality for 𝑊 . Before pub-
lication, a third session with event time inside 𝑊 arrives as a docu-
mented correction and is not cited. A digest of the two cited records
is unchanged, but CORG’s session selector result and correction
state change, so approval is held for re-evaluation. If the approval
register is complete and no matching authorization exists, the new
witness may change the verdict to VIOLATED. An unrelated Fab-B
record outside the selector leaves the approved dependency finger-
print unchanged. These are test cases, not observed results.
4.4 Eligible graph assisted retrieval
The normative store contains authorized, versioned policies and
reference material. The operational store contains events, config-
urations, asset records and approvals. Proposed node types are
Site, Asset, Zone, Conduit, Requirement, PolicyVersion, Evidence,
Claim and Approval. Typed relationships include belongs_to, con-
nects_to, governed_by, observed_at and cites. Every factual rela-
tionship retains a source reference and validity interval. An LLM-
extracted relation remains unverified until corroborated; it cannot
silently become authoritative topology or policy.
Retrieval first applies principal entitlement, asset/site scope, in-
terval and policy applicability. It then combines text candidates
with bounded graph traversal, for example a session-to-account-to-
approval or asset-to-zone-to-approved-path chain. A default max-
imum of two hops and a fixed context-token budget are proposed
starting settings, tuned only on development data. Every expan-
sion applies the same access and scope filters. Candidates are dedu-
plicated and ranked; graph paths must resolve to registered source
evidence. Standard or policy descriptions explain what is required
and cannot serve alone as proof that a facility implemented it.
4.5 Structured local model analysis
The local model emits a constrained schema with a proposed de-
cision, atomic claims, citations as evidence_id plus source offsets,
missing evidence and contradictions. Inference records checkpoint
revision, tokenizer, quantization, prompt version, decoding param-
eters and retrieved-bundle digest. The model can read the autho-
rized bundle but cannot define Σ𝑟 , modify the registry or policy,
approve its analysis or invoke OT control actions. Invalid JSON
and unsupported scope are explicit failures.
4.6 Three-valued judgment and proof
certificate
The input layer inspects both the question and retrieved payloads,
distinguishes quoted data from instructions, limits available tools
and records injection flags. Detector checkpoint and threshold
must be specified and independently calibrated; detection alone
does not establish release safety. Benign discussion of an attack
remains valid task data.
The output layer first performs deterministic binding and clo-
sure checks, then evaluates semantic support of each atomic claim
against eligible spans using an independent verifier. Exact identi-
fiers, intervals and numerical statements use explicit comparison.
A mean support score cannot hide one decisive unsupported claim.
Let 𝑠 (𝑞0; 𝜏) = 1 only when the positive proof branch and every
coverage premise it uses are discharged; let 𝑣 (𝑞0; 𝜏) = 1 only when
a decisive violation branch and any absence premise it uses are
discharged. For each clause CORG retains both bits, so (1, 1) is a
conflict, not a pass. Its exposed verdict is
𝑦 (𝑞0; 𝜏) =
8>>> <
>>>:
SUPPORTED (𝑠, 𝑣) = (1, 0);
VIOLATED (𝑠, 𝑣) = (0, 1);
INSUFFICIENT (𝑠, 𝑣) ∈ {( 0, 0), (1, 1)} .
(1)
Coverage is checked only where the proof branch requires it;
a direct counterexample need not wait for unrelated collectors.
Conversely, a missing event or approval is never a decisive wit-
ness without closure and coverage of the relevant authoritative
source. Each factual assertion in a violation or insufficiency find-
ing must also have eligible support. Human review cannot waive
the positive-evidence predicate.
The auditable certificate 𝜋 (𝑞0; 𝜏) records obligation version
and applicability, the proof branch, claim-to-span witnesses,
unresolved gaps, every selector and scoped result digest, interval-
scoped coverage claim, cited raw-byte digests and applicable
policy/topology revisions. Its canonical decision content 𝜋 ∗ ex-
cludes the changing snapshot timestamp. Global source cursors
and raw heartbeat timestamps remain in a separate audit trace;
they do not change the approved digest unless they alter a re-
quired scoped predicate. Semantic thresholds are selected only on
an OT development set and reported with risk–resolution curves;
an uncalibrated support score is not a hallucination probability.
Semi-Shield: Obligation-Closed OT Evidence SAC ’27, April 5–9, 2027, Gwangju, South Korea
MiniCheck and any other NLI checkpoint are separate, frozen
configurations.
4.7 Approval-bound publication and
conditional invariant
A failed or uncertain candidate enters review. At most one bounded
revision may address flagged defects, after which all checks run
again. Fallback can state a qualified gap but cannot directly release
an affirmative judgment. Guard outage or disabled safety profile
routes to review; every output path uses the same publication API.
The reviewer sees the exact rendered report and canonical de-
pendency fingerprint
𝐹𝑎 = 𝐻  𝜋 ∗ (𝑞0; 𝜏𝑎), 𝑅(𝑞0; 𝜏𝑎) , 𝐷 𝑎 = 𝐻 (report, 𝑞0, 𝐹𝑎, config) .
(2)
An authenticated, expiring, one-use approval binds 𝐷𝑎; 𝜏𝑎 is
retained in the audit log, not in 𝐹𝑎. At release snapshot 𝜏𝑟 ≥ 𝜏𝑎,
Reporter reruns the same selectors over event interval 𝑊 with
ingest cutoff 𝜏𝑟 , reloads cited raw bytes and normalized cover-
age claims, checks current applicability and governing versions,
and recomputes 𝐹𝑟 . It releases only if 𝐹𝑟 = 𝐹𝑎, exact report
bytes/configuration match, the token is valid and the approved
proof branch still verifies. A stochastic model need not be rerun
when the canonical inputs are unchanged. A relevant correction
or new record, loss intersecting 𝑊 , changed cited bytes, applicable
policy/topology change or altered report invalidates approval
and requires fresh assessment/review. An unrelated record, later
heartbeat or version update outside the obligation’s selectors does
not. An explicitly labeled archival “known as of 𝜏𝑎” report is a
separate mode and must not assert release-time status.
All gate inputs—evidence, coverage, applicable policy and topol-
ogy, entitlement, approval state and persisted report—are material-
ized in one versioned local authority view. Every ingestion or cor-
rection commits there before it becomes visible to assessment or re-
lease. The release uses a serializable transaction that validates pred-
icate ranges (or an equivalent selector version fence maintained
transactionally), checks 𝐹𝑟 and consumes approval while persist-
ing the exact report. That commit is the local linearization point.
Snapshot isolation without phantom protection is insufficient. The
public AGT implementation supplies action-bound approval, but a
durable store and this atomic publisher must be implemented and
tested for every comparison arm [ 21]. Changes still only in exter-
nal systems, or later corrections, cannot be covered by this release
invariant; feed lag and post-release retraction are measured and re-
ported separately. If a deployment cannot provide the local atomic
boundary, release is held rather than claimed safe.
The conditional invariant tested is: every RELEASED affirma-
tive report has an eligible proof and valid human approval for the
exact state of the local authority view at its release commit. It de-
pends on trusted acquisition and policy sources, sound obligation
templates, complete selectors, checker accuracy and atomic local
publication; hashes do not prove that physical events occurred.
AGT could enforce the same digest if supplied CORG’s 𝑅, so the
evaluated method contribution is derivation and revalidation of 𝑅,
not a new approval primitive.
Algorithm 1: CORG assessment and release (proposed)
1. Assess: instantiate 𝑂𝑟 ; execute every 𝜎 ∈ Σ𝑟 over 𝑊 on local snap-
shot 𝜏𝑎; validate spans, coverage and proof branches; emit 𝑦, 𝜋, 𝑅, 𝐹𝑎
and a frozen report.
2. Approve: show the exact report and dependencies; bind one-use
approval to 𝐷𝑎 = 𝐻 (report, 𝑞0, 𝐹𝑎, config).
3. Release: in a serializable local transaction at 𝜏𝑟 , rerun Σ𝑟 over the
same 𝑊 with current ingest cutoff, validate predicate reads, applicable
versions and approval; persist only if 𝐹𝑟 = 𝐹𝑎 and the exact binding
matches. Any change routes to REVIEW_REQUIRED.
Figure 2: CORG binds publication to obligation-derived
predicate reads, including empty result sets and observation
coverage.
4.8 Optional small model fine tuning
Task-specific small-model adaptation is optional. Supervised exam-
ples could teach scoped assessment, faithful citations, contradic-
tion detection and explicit insufficiency using authorized, indepen-
dently reviewed training groups. Current facility facts remain re-
trieval evidence, not memorized model authority. Training config-
uration and a base-versus-tuned factorial experiment are required
before claiming an improvement; large-model parity requires an
additional matched comparator and a prespecified statistical crite-
rion.
5 EV ALUATION PROTOCOL AND UNFILLED
RESULTS
5.1 Planned benchmark and annotation
The proposed Semi-Shield OT Evidence Evaluation Set targets
120 independent held-out scenario groups (20 per monitoring
family in Table 1) and 30 separate development groups. Split
by scenario, source-trace family and template before producing
four paired variants per group: clean, decisive evidence removed,
scope-mismatched evidence and indirect injection in untrusted
content. The target is 480 test packets and 120 development pack-
ets, not 480 independent incidents. These are planned counts;
observed counts appear in Table 2 (held-out run). Any fine-
tuning requires additional, disjoint training/validation groups.
Test labels, threshold choices and near-duplicate source traces
must not leak into development.
Clean cases include both supported and violated obligations.
Removal and scope mutations are re-labeled, not automatically
marked INSUFFICIENT; a remaining counterexample may still
prove violation. Injection preserves task facts and gold wher-
ever possible, with a frozen attacker objective. Two qualified
OT/security reviewers independently annotate applicability, ver-
dict, all minimal sufficient evidence sets, exact spans, selector
eligibility, coverage and potential counterexamples before observ-
ing system output; an adjudicator resolves disagreements. Report
class counts, raw agreement, Cohen’s 𝜅, and evidence-set/span
agreement. Distinguish author-created simulation, laboratory,
sanitized historical and adapted records. At least one indepen-
dently sourced or expert-authored family should test transfer
SAC ’27, April 5–9, 2027, Gwangju, South Korea
beyond the generation templates; synthetic data alone do not
establish production validity.
Each case records event and ingest time, raw digest, site, asset
and zone, applicable policy and topology versions, source selectors,
coverage heartbeats, loss and finality, cited and uncited records,
exact report and approval state. Gold publication decisions are an-
notated separately from gold assessment labels. The public release
should contain schema, redacted or author-owned evidence, gold
rubric, split manifest, case traces and scoring code; restricted real
records may be evaluated under a documented controlled protocol.
5.2 Baselines and matched comparisons
Freeze model/checker revisions, context and output budget, cor-
pus/index snapshot, decoding, top- 𝑘, thresholds and input detec-
tor. B2 and B3 receive identical authoritative records, entitlement
and metadata filters. The primary RQ1 comparison feeds B4 and
B5 the same draft and retrieved bundle , with zero revisions; CORG
may independently run its required closure selectors. B0 receives
the same pre-extracted draft facts, whereas B0-full receives all of
B5’s authoritative closure-query outputs and coverage facts. This
separates rule inference from evidence-access advantage. Then run
complete pipelines with identical permitted retry budget and dis-
tinguish pre-human from corrected reports. An optional direct-
reading B1 can contextualize retrieval cost.
For RQ2, keep the same writer API, durable approval store, se-
rializable local authority view, enforcement point, candidate re-
port and human approval process, and use the public AGT action-
bound approval implementation [ 21] as the execution substrate.
Pin its library revision. Compare AGT-cited (exact report and cited-
evidence digests), AGT-global (report and entire corpus revision),
and AGT+CORG (report and obligation-derived predicate read set).
AGT-cited is not characterized as an AGT defect: it receives a de-
liberately narrower dependency description. If AGT is supplied
CORG’s full read-set digest and requery logic, it should enforce
the same gate; this parity configuration verifies the integration.
Test relevant new counterevidence, loss of collector coverage, post-
finality corrections inside 𝑊 , policy/topology updates, unrelated
record additions, stale/replayed approvals, guard outage and con-
current publishers. Record the actual persisted report, not just a pro-
posed tool call. Measure source-feed lag and affected-report retrac-
tion after a post-release correction separately.
5.3 Metrics and statistical analysis
The primary judgment endpoints are false-affirmative rate (FAR:
predicted SUPPORTED among all gold non-SUPPORTED tasks)
and answerable-case resolution (correct determinate decisions
among all gold SUPPORTED or VIOLATED tasks). Report the two
jointly at a threshold/operating point chosen on development data
and as a risk–resolution curve; a system that rejects everything
cannot claim useful improvement. Also report false VIOLATED
rate, three-class macro-F1, complete minimal-evidence-set recall,
selector recall against independently enumerated authoritative
records, claim support precision (eligible joint support among
all substantive claims, including uncited ones), and claim–span
citation validity. Unsupported eligible positives are reported both
per eligible positive draft and per all tasks.
For macro-F1, retain every task. Invalid or blocked output re-
ceives an abstention symbol, contributes a false negative to its gold
class, and is never credited as INSUFFICIENT. Report pre-gate F1,
post-gate eligible resolution, invalid-output and review rates sepa-
rately. Human corrections are excluded from autonomous quality
metrics and reported as a separate workflow outcome.
The primary release endpoint is unauthorized actual release
rate: stale, unapproved or mismatched reports persisted divided by
all adversarial/relevant-mutation release attempts. Complement it
with legitimate release rate on unchanged approved cases, selec-
tive release rate after irrelevant updates, unnecessary re-review
rate and reviewer minutes per case. Injection success requires
the frozen attacker goal to occur in the persisted output; report
normal-task utility under attack. A zero observed bypass count
receives its exact binomial upper confidence bound, not a claim of
zero risk.
Use paired 95% confidence intervals for method differences by
resampling complete scenario groups, retaining all variants and
repeated runs together; 2,000 bootstrap draws with a fixed seed
are the planned analysis. Prespecify B5 versus B4 for RQ1 and
AGT+CORG versus AGT-cited/global for RQ2; B3 versus B2 is
a separate retrieval comparison. Small stress suites demonstrate
mechanisms rather than estimate rare failure rates. Report warm
p50/p95 latency across retrieval, closure, checker, revision and
serialization, cold loading separately, peak RAM/VRAM, calls,
index build/update cost and reviewer time on fixed hardware.
5.4 Security stress tests and ablations
Include independently defined stale-policy/topology, fabricated-
span, simulated-as-production and contradictory-source cases.
Multi-asset records must include eligible outer metadata but an
ineligible cited span, interval or edge, since trivial mismatches
are already removed by shared retrieval filters. Mutation tests
edit/delete/reorder cited bytes, swap attachments, add uncited
relevant counterevidence, remove heartbeat coverage and add
irrelevant records. Publication tests exercise fallback, AI review,
guard outage, edge mode, replayed approval and two concurrent
publishers without a valid new human token. A whole-chain
rewrite is reported as undetectable unless an independent check-
point is actually present.
Single-removal ablations disable (one at a time) predicate clo-
sure while retaining cited-bundle hashes, coverage validation,
scope/span validation, semantic checking, policy/topology ver-
sion binding or release-time requery. Each ablation is paired with
cases exercising that dependency. Graph traversal and retrieved-
content injection screening are separately ablated so their effects
are not credited to CORG. If a tuned model exists, cross base/tuned
model with CORG absent/present while keeping retrieval fixed;
no fine-tuning result is presumed.
5.5 Public resources and domain transfer
Public auxiliary probes use FEVEROUS’s official three-way and
complete-evidence-set scoring, ContractNLI’s NLI/span task and
ACE’s multi-clause labels on their original splits [ 17, 18, 19].
They are reported separately, never pooled with OT gold; Con-
tractNLI’s NotMentioned and ACE’s Not-Applicable cannot be
Semi-Shield: Obligation-Closed OT Evidence SAC ’27, April 5–9, 2027, Gwangju, South Korea
Table 2: Dataset composition (measured held-out run).
Partition Target groups Actual groups / packets / S,V,I / agreement
Development 30 / / /
Held-out OT 120 120 / 522 / 240,42,240 / rule-gold
Release mutations paired with held-out trials; relevant; irrelevant 36+36 trials / see Table 5 / —
Transfer family to specify / / / (not run)
Table 3: Matched assessment baselines. All share a corpus and entitled metadata filters.
ID Configuration Comparison purpose
B0 OPA/Rego obligation rules Deterministic rules on the same draft’s extracted facts; may abstain
B0-full OPA/Rego with closure outputs Strong rules comparator with the same authoritative selector results and coverage
facts as B5
B2 Local model plus filtered
BM25/vector RAG
Cited, insufficiency-aware ordinary retrieval
B3 B2 plus typed graph traversal Isolates relationship retrieval
B4 B3 plus MiniCheck grounding Generic verifier with the same checker and threshold as B5
B5 B4 plus CORG closure/certificate Tests obligation-derived source, coverage and version semantics
silently mapped to OT INSUFFICIENT. RAGTruth probes the
grounding checker, not control effectiveness [ 6]. Loopjacking’s
public reproduction archive supplies approval-substitution re-
gression cases and AgentDojo supplies agent injection tasks
where tools are actually executed [ 22, 7]. Neither is an OT gold
benchmark. HAI is downloadable OT process context; SWaT
requires an access request and redistribution restrictions [ 8, 9].
Their attack/normal labels cannot become compliance labels.
ReCon is related documentary-compliance work, not a directly
downloadable operational benchmark [ 14].
A cross-domain extension would use at least 30 independently
annotated water or power OT groups with new templates and poli-
cies, keeping all training and threshold selection in manufacturing.
Test-time retrieval may supply the new domain’s factual and pol-
icy records but never gold assessments. Report per-domain per-
formance and degradation. Fab-A versus Fab-B is cross-site, not
cross-domain; synthetic transfer must be described as synthetic.
6 RESULTS AND ANALYSIS
Measured results were produced with repository `ot_test` (code/eval, scripts/run_sac_corg_evaluation.py). Gold labels are rule-generated from ot/ syslog with clean, removed, scope-mismatch and injection variants; expert dual annotation remains future work.
On 522 held-out packets, CORG (B5) achieved FAR 0.00 versus B4 FAR 40.78 (Δ -0.407801 [-0.4545, -0.3601]). Resolution for B5 matched B0-full at approximately 50.35%, avoiding a reject-all operating point. Release binding: AGT-cited permitted unauthorized release on all relevant counterevidence trials in simulation, while AGT+CORG permitted none (Table 5). Fab-A versus Fab-B cross-site strata are described in Section 5.1 of the implementation note data/eval_sac/results.json (cross-site, not cross-domain).
Public-probe scores (FEVEROUS, ContractNLI, ACE) were not pooled with OT gold. Risk–resolution curves and annotation disagreement are not yet included.
7 DISCUSSION AND LIMITATIONS
The scope is semiconductor OT network and management evi-
dence. Device-login analysis does not demonstrate PLC process
safety, and a scoped finding is not an ISO certificate. CORG’s
obligation templates and closure selectors require expert au-
thorship and can omit a relevant source. Coverage watermarks
express trusted collection assumptions, not physical truth. A
forged authenticated source, faulty clock, compromised reviewer
or non-atomic external export can invalidate the claimed bound-
ary. Semantic verifiers can reject valid paraphrases or accept
unsupported combinations. AGT with the same CORG dependen-
cies should be functionally equivalent at the approval layer; any
advantage must be attributed to obligation/read-set construction
or its measured efficiency. Synthetic evidence, limited vendors
and small transfer strata constrain external validity, while human
review can introduce error and delay.
The evidence corpus must have documented rights for the
intended retrieval, training and release uses. ISO’s official pages
state AI-use restrictions outside specified exceptions or permission
[1], [3]. A purchased standard should not be assumed to authorize
model training or RAG. The proposed public release can instead
use author-written organizational policies, permitted government
guidance and reviewed task annotations, without redistributing re-
stricted standard text. Source acquisition permissions and dataset
rights must be recorded before experiments.
8 CONCLUSION
This draft defines CORG as a single, testable link between OT ev-
idence obligations and approval-bound publication. Its central hy-
potheses concern (i) three-valued judgments supported by com-
plete, scoped observation and (ii) selective invalidation of approval
when decision-relevant evidence changes. Strong public approval
gates and generic grounding methods make those comparisons
necessary. The proposed paired benchmark and release mutations now include observed Table 4–6 entries; CORG (B5) reduced FAR to zero on held-out gold in this run.
SAC ’27, April 5–9, 2027, Gwangju, South Korea
Table 4: OT judgment results (%). FAR lower is better; resolution, F1 and evidence-set recall higher are better.
Method FAR Resolution Macro-F1 Evidence recall p95 s
B0 57.45 85.11 47.14 — —
B0-full 0.00 50.35 65.33 — —
B2 43.26 63.83 44.36 — —
B3 40.78 52.13 43.55 — —
B4 40.78 52.13 43.55 — —
B5 0.00 50.35 65.33 — —
Table 5: Actual report-release results (measured rates).
Approval binding Unauthorized release Unchanged valid release Irrelevant-update release Re-review
AGT-cited 100.00 — 100.00 —
AGT-global 0.00 — 100.00 —
AGT+CORG 0.00 — 100.00 —
AGT+same_R 0.00 — 100.00 —
Table 6: Interpretation guide (observed effects; paired bootstrap 2000 draws).
Contrast Supportive pattern If absent Observed Δ [CI]
B5 vs B4 FAR lower with CORG at comparable resolution — -0.407801 [-0.4545, -0.3601]
B3 vs B2 graph retrieval improves answerable resolution — -0.117021 [-0.1625, -0.0742]
AGT+CORG vs AGT-cited positive delta means CORG blocks more unsafe releases — 1.0 —
REFERENCES
[1] ISO/IEC, ISO/IEC 27001:2022, Information security management systems — Re-
quirements, 2022. Official overview: https://www.iso.org/standard/27001 (ac-
cessed 11 October 2026).
[2] K. Stouffer et al., Guide to Operational Technology Security, NIST SP 800-82
Rev. 3, 2023, doi:10.6028/NIST.SP.800-82r3. https://csrc.nist.gov/pubs/sp/800/82/
r3/final. This draft cites the final Rev. 3; Rev. 4 is currently an initial public draft.
[3] ISO/IEC, ISO/IEC 27002:2022, Information security controls, 2022. https://www.
iso.org/standard/75652.html. Control references identify context; standard text
is not reproduced as a training corpus.
[4] Microsoft, GraphRAG documentation, Local Search. https://microsoft.github.io/
graphrag/query/local_search/ (accessed 11 October 2026).
[5] L. Tang, P. Laban, and G. Durrett, “MiniCheck: Efficient Fact-Checking
of LLMs on Grounding Documents, ” EMNLP, pp. 8818–8847, 2024,
doi:10.18653/v1/2024.emnlp-main.499. https://aclanthology.org/2024.emnlp-
main.499/.
[6] C. Niu et al., “RAGTruth: A Hallucination Corpus for Developing Trust-
worthy Retrieval-Augmented Language Models, ” ACL, pp. 10862–10878, 2024,
doi:10.18653/v1/2024.acl-long.585. https://aclanthology.org/2024.acl-long.585/.
[7] E. Debenedetti et al., “AgentDojo: A Dynamic Environment to Evaluate Prompt
Injection Attacks and Defenses for LLM Agents, ” NeurIPS Datasets and Bench-
marks, 2024. https://agentdojo.spylab.ai/.
[8] ICSdataset, HAI Security Dataset, official repository and technical documenta-
tion. https://github.com/icsdataset/hai (accessed 11 October 2026).
[9] SUTD iTrust, SWaT dataset characteristics. https://www.sutd.edu.sg/itrust/
itrust-labs/datasets/dataset-characteristics/swat/ (accessed 11 October 2026).
[10] NIST, Log file integrity checking, Computer Security Resource Center glossary.
https://csrc.nist.gov/glossary/term/log_file_integrity_checking.
[11] NIST, Assessing Security and Privacy Controls in Information Systems and Or-
ganizations, SP 800-53A Rev. 5, 2022. https://csrc.nist.gov/pubs/sp/800/53/a/r5/
final.
[12] NIST, OSCAL Assessment Results Model, official documentation. https:
//pages.nist.gov/OSCAL/learn/concepts/layer/assessment/assessment-results/
(accessed 12 October 2026).
[13] R. Bolton et al., “Multi-Stage Retrieval for Operational Technology Cyber-
security Compliance Using Large Language Models: A Railway Casestudy, ”
arXiv:2504.14044, 2025. https://arxiv.org/abs/2504.14044.
[14] R. Negi et al., “ReCon: A Resource-Constrained Benchmark for LLM-
Based Cybersecurity Compliance Across Ingestion and Retrieval Pipelines, ”
arXiv:2607.22885, 2026. https://arxiv.org/abs/2607.22885.
[15] F. Ikhwantri et al., “An Agentic RAG and Evaluation Framework for Assurance
Case Generation: Industrial Use Case for the EU Cyber Resilience Act Compli-
ance, ” arXiv:2608.19509, 2026.https://arxiv.org/abs/2608.19509.
[16] M. T. Moghaddam, “Compliance-by-Construction Argument Graphs: Using Gen-
erative AI to Produce Evidence-Linked Formal Arguments for Certification-
Grade Accountability, ” arXiv:2604.04103, 2026.https://arxiv.org/abs/2604.04103.
[17] FEVER, FEVEROUS 2021 Shared Task: structured and unstructured evidence ver-
ification, official task and baseline. https://fever.ai/2021/task.html (accessed 12
October 2026).
[18] Y. Koreeda and C. Manning, “ContractNLI: A Dataset for Document-level Natu-
ral Language Inference for Contracts, ” Findings of EMNLP, pp. 1907–1919, 2021.
https://aclanthology.org/2021.findings-emnlp.164/.
[19] A. Singh et al., “COMPACT: Building Compliance Paralegals via Clause Graph
Reasoning over Contracts, ” EACL, pp. 8081–8112, 2026.https://aclanthology.org/
2026.eacl-long.377/.
[20] Open Policy Agent, “Agent Tool Calling, ” official documentation. https://www.
openpolicyagent.org/docs/agent-tool-calling (accessed 12 October 2026).
[21] Microsoft, “Action-Bound, Fail-Closed Approval Protocol, ” Agent Gov-
ernance Toolkit ADR-0030 and public .NET implementation, 2026.
https://github.com/microsoft/agent-governance-toolkit/blob/main/
docs/adr/0030-action-bound-approval-protocol.md ; implementation:
https://github.com/microsoft/agent-governance-toolkit/blob/main/agent-
governance-dotnet/README.md#action-bound-approval-chains .
[22] A. A. Kumar, “Loopjacking: Hijacking Human-in-the-Loop Approval, ”
arXiv:2609.21081, 2026. https://arxiv.org/abs/2609.21081.
[23] S. Khan, “Stop Means Stop: Measuring and Repairing the Enforcement Gap in
Agent-Framework Control Primitives, ” arXiv:2607.14166, 2026.https://arxiv.org/
abs/2607.14166.
[24] J. He and D. Yu, “Stored Is Not Supported: Typed Provenance and Assertion
Guardrails for Persistent AI Agents, ” arXiv:2609.02127, 2026. https://arxiv.org/
abs/2609.02127.