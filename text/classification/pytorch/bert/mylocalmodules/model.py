def get_bert_model(method, pre_trained, num_labels=None, return_dict=False):
    '''
    bert model 을 구하는 함수

    parameters
    ----------
    pre_trained: str
        모델에 적용할 pre-trained weight

    num_labels: int
        모델이 분류할 라벨 갯수 (BertForSequenceClassification 용)

    return_dict: bool
        dict 를 얻어낼 것인지 여부. (BertModel 용)
    returns
    -------
    model: bert model
        bert 모델
    '''
    if method == 'BertModel':
        from transformers import BertModel
        model = BertModel.from_pretrained(pre_trained, return_dict=False)
    if method == 'BertForSequenceClassification':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(pre_trained, 
                                                              num_labels=num_labels)
    return model