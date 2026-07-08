"""
feature_engineering.py — shared logic between training (Section 11) and serving (app.py),
so the two can never silently drift apart.
"""
import numpy as np
import pandas as pd

EDU_ORDER = {'No education': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}
WEALTH_ORDER = {'Poorest': 0, 'Poorer': 1, 'Middle': 2, 'Richer': 3, 'Richest': 4}
PARTNER_EDU_ORDER = {'No partner': -1, 'No education': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}
AGE_GROUP_ORDER = {'15-19': 0, '20-24': 1, '25-29': 2, '30-34': 3,
                    '35-39': 4, '40-44': 5, '45-49': 6}

ARID_COUNTIES = ['Turkana', 'West Pokot', 'Mandera', 'Wajir', 'Garissa',
                  'Marsabit', 'Samburu', 'Isiolo', 'Tana River']

REGION_MAP = {
    'Mombasa': 'Coast', 'Kwale': 'Coast', 'Kilifi': 'Coast', 'Tana River': 'Coast',
    'Lamu': 'Coast', 'Taita Taveta': 'Coast',
    'Garissa': 'North Eastern', 'Wajir': 'North Eastern', 'Mandera': 'North Eastern',
    'Marsabit': 'Eastern', 'Isiolo': 'Eastern', 'Meru': 'Eastern', 'Tharaka-Nithi': 'Eastern',
    'Embu': 'Eastern', 'Kitui': 'Eastern', 'Machakos': 'Eastern', 'Makueni': 'Eastern',
    'Nyandarua': 'Central', 'Nyeri': 'Central', 'Kirinyaga': 'Central',
    "Murang'a": 'Central', 'Kiambu': 'Central',
    'Turkana': 'Rift Valley', 'West Pokot': 'Rift Valley', 'Samburu': 'Rift Valley',
    'Trans Nzoia': 'Rift Valley', 'Uasin Gishu': 'Rift Valley', 'Elgeyo Marakwet': 'Rift Valley',
    'Nandi': 'Rift Valley', 'Baringo': 'Rift Valley', 'Laikipia': 'Rift Valley',
    'Nakuru': 'Rift Valley', 'Narok': 'Rift Valley', 'Kajiado': 'Rift Valley',
    'Kericho': 'Rift Valley', 'Bomet': 'Rift Valley',
    'Kakamega': 'Western', 'Vihiga': 'Western', 'Bungoma': 'Western', 'Busia': 'Western',
    'Siaya': 'Nyanza', 'Kisumu': 'Nyanza', 'Homa Bay': 'Nyanza', 'Migori': 'Nyanza',
    'Kisii': 'Nyanza', 'Nyamira': 'Nyanza',
    'Nairobi': 'Nairobi',
}


def engineer_features(record: dict) -> pd.DataFrame:
    """Reproduces Section 11\'s feature engineering for a single respondent record
       submitted to the API, returning the 36 model-ready columns the LightGBM
       pipeline expects."""
    r = dict(record)

    has_given_birth = int(r.get('age_first_birth', 0) not in (None, 0, np.nan))
    r['has_given_birth'] = has_given_birth
    r['age_first_birth'] = r.get('age_first_birth') or 0

    r['education_level_ord'] = EDU_ORDER.get(r.get('education_level'), 0)
    r['wealth_index_ord'] = WEALTH_ORDER.get(r.get('wealth_index'), 0)
    r['partner_education_ord'] = PARTNER_EDU_ORDER.get(r.get('partner_education', 'No partner'), -1)
    r['age_group_ord'] = AGE_GROUP_ORDER.get(r.get('age_group'), 0)

    gap = r['partner_education_ord'] - r['education_level_ord']
    r['education_gap'] = 0 if r['partner_education_ord'] == -1 else gap
    r['has_partner'] = int(r.get('partner_education', 'No partner') != 'No partner')

    household_size = max(r.get('household_size', 1), 1)
    children_ever_born = r.get('children_ever_born', 0)
    living_children = r.get('living_children', 0)

    r['child_density'] = children_ever_born / household_size
    r['surviving_ratio'] = (living_children / children_ever_born
                             if children_ever_born > 0 else 1.0)
    r['child_loss'] = children_ever_born - living_children

    r['is_in_union'] = int(r.get('union_status') == 'Currently in union')
    r['is_married_or_union'] = int(r.get('marital_status') in ('Married', 'Living together'))

    r['urban'] = int(r.get('residence_type') == 'Urban')
    r['employed'] = int(r.get('currently_working') == 'Yes')
    r['female_hh_head'] = int(r.get('household_head_sex') == 'Female')
    r['had_pregnancy_loss'] = int(r.get('pregnancy_loss') == 'Yes')

    r['age_at_first_birth_gap'] = (
        r.get('age', 0) - r['age_first_birth'] if has_given_birth else 0
    )

    r['arid_county'] = int(r.get('county') in ARID_COUNTIES)
    r['region'] = REGION_MAP.get(r.get('county'), 'Nairobi')

    cols = ['age', 'age_group', 'county', 'residence_type', 'education_level', 'religion',
            'household_size', 'household_head_sex', 'wealth_index', 'children_ever_born',
            'age_first_birth', 'living_children', 'pregnancy_loss', 'marital_status',
            'union_status', 'partner_education', 'currently_working', 'has_given_birth',
            'education_level_ord', 'wealth_index_ord', 'partner_education_ord', 'age_group_ord',
            'education_gap', 'has_partner', 'child_density', 'surviving_ratio', 'child_loss',
            'is_in_union', 'is_married_or_union', 'urban', 'employed', 'female_hh_head',
            'had_pregnancy_loss', 'age_at_first_birth_gap', 'arid_county', 'region']

    return pd.DataFrame([{c: r.get(c) for c in cols}])
