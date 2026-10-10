"""Selection-to-writing propagation and local excerpt construction, without API calls."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

from utils.evidence_material import EvidenceSelection, writing_evidence
from utils.llm_schemas import SelectedIdsResponse
from streaming.listening_evidence import EvidenceState
from streaming.body_task import BodyTask
from utils.prompts.speech_revision import revision_prompt


def test_native_selection_preserves_analysis_and_list_interface_without_mutation():
    from ouragents import TreeDebater
    candidates = [dict(id=str(i), content='Funding supports access.', raw_content='FULL') for i in range(12)]
    original = copy.deepcopy(candidates)
    owner = SimpleNamespace(motion='Fund access', side='for', helper_client=Mock(return_value=[
        SelectedIdsResponse(selected_ids=['2'], analysis={'2': 'Supports funding feasibility, not guaranteed access.'})]))
    result = TreeDebater._select_revision_evidence(owner, 'Funding enables access.', '', candidates, stage='opening')
    assert isinstance(result, list) and result == [candidates[2]]
    assert result.analysis == {'2': 'Supports funding feasibility, not guaranteed access.'}
    assert owner.helper_client.call_count == 1 and candidates == original
    assert copy.deepcopy(result).analysis == result.analysis


def test_cache_retains_analysis_for_selected_ids_without_trusting_returned_source_text():
    evidence = [dict(id='fund', title='Budget report', content='Funding increased access. Results apply only to this trial.')]
    selector = Mock(return_value=EvidenceSelection([dict(evidence[0], content='INVENTED')],
                    {'fund': 'Supports the funding step, not universal success.', 'unknown': 'Wrong ID'}))
    state = EvidenceState()
    result = state.select(selector, 'Funding increased access.', '', evidence, stage='opening')
    assert result == evidence and 'INVENTED' not in str(result)
    reused = state.select(selector, 'Funding increased access.', 'No changes', evidence, stage='opening')
    assert selector.call_count == 1 and reused.analysis == result.analysis
    cards = writing_evidence(reused, 'Funding increased access.')
    assert cards[0]['selection_reason'] == result.analysis['fund']
    assert 'only to this trial' in cards[0]['content']


def test_local_excerpts_keep_neighbors_and_drop_unrelated_documents_and_metadata():
    irrelevant = 'Ocean currents shape fisheries. ' * 300
    content = irrelevant + '\nFunding increased access in the trial. However, it did not improve outcomes elsewhere.\n' + irrelevant
    selection = EvidenceSelection([
        dict(id='a', title='Funding trial', content=content, source='Institute', date='2024', raw_content='RAW', numbers=['99%']),
        dict(id='b', title='Fisheries', content=irrelevant)], {'a':'Supports trial access only.'})
    cards = writing_evidence(selection, 'Funding can increase access.', n_words=500)
    assert len(cards) == 1 and cards[0]['id'] == 'a'
    assert cards[0]['content'] in content
    assert 'did not improve outcomes elsewhere' in cards[0]['content']
    assert cards[0]['date'] == '2024'
    assert 'RAW' not in json.dumps(cards) and '99%' not in json.dumps(cards)
    assert len(json.dumps(cards)) < len(json.dumps(list(selection))) / 5


def test_duplicate_sources_merge_reasons_and_ids():
    evidence = EvidenceSelection([
        dict(id='a', title='Funding trial', content='Funding increased access.'),
        dict(id='b', title='Funding trial', content='Funding improved rural access.')],
        {'a': 'Supports access.', 'b': 'Supports rural access only.'})
    cards = writing_evidence(evidence, 'Funding improves access.')
    assert len(cards) == 1
    assert set([cards[0]['id']] + cards[0]['also_selected_ids']) == {'a', 'b'}
    assert 'Supports access.' in cards[0]['selection_reason']
    assert 'Supports rural access only.' in cards[0]['selection_reason']


def test_missing_metadata_is_not_invented_and_no_fixed_evidence_count():
    evidence = EvidenceSelection([dict(id=str(i), content=f'Funding result {i}.') for i in range(15)])
    cards = writing_evidence(evidence, 'Funding results.', n_words=1000)
    assert len(cards) == 15
    assert all(not e['source'] and 'date' not in e for e in cards)


def test_speculative_and_committed_prompts_keep_identical_annotated_material():
    from utils import speech_length
    evidence = EvidenceSelection([dict(id='a', title='Trial', content='Funding increased access.')], {'a': 'Supports access.'})
    task = BodyTask.create(motion='Fund access',side='for',stage='opening',history=[],
                          prefix='We support access.',framework={},draft='Funding can increase access.')
    early = task.revision_prompt('Explain access.', 60, evidence)
    committed = revision_prompt(motion='Fund access',side='for',stage='opening',statement=task.tail,
        feedback=task.guidance('Explain access.'), allocation_plan=task.allocation,evidence=evidence,
        prefix='We support access.',n_words=speech_length.draft_word_budget(60))
    assert early == committed and 'Supports access.' in committed
    assert 'WHOLE LOGIC' not in committed and 'Selection reasons are fallible writing hints, never evidence themselves.' in committed


def test_decimal_findings_remain_verbatim():
    content = 'Funding improved access by 12.5 percent. This was limited to one district.'
    cards = writing_evidence([dict(id='a',content=content)], 'Funding improves access.')
    assert cards[0]['content'] == content


def test_material_covers_distinct_current_passages_before_repeating_one():
    evidence = [dict(id='a',content='Funding budget access increased.'),
                dict(id='b',content='Funding budget access expanded.'),
                dict(id='c',content='Hospitals infection mortality declined.')]
    cards = writing_evidence(evidence, 'Funding budget access.\n\nHospitals infection mortality.')
    assert [c['id'] for c in cards[:2]] == ['a', 'c']
