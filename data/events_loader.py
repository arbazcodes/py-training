import torch
from torch.utils.data import Dataset


class EventsLoader(Dataset):
    def __init__(
        self,
        batch_size,
        src_seq_length,
        tgt_seq_length,
        id_category_size
    ):
        self.batch_size = batch_size
        self.src_sequence_length = src_seq_length
        self.tgt_sequence_length = tgt_seq_length

        self.encoder_streams = [
            'EventType',
            'RecipientRecordId',
        ]

        self.initial_category_fields = {
            'EventType': 10,
            'ActorRecordId': int(id_category_size),
            'RecipientRecordId': int(id_category_size),
            'LocationRecordId': int(id_category_size),
            'CostCodeRecordId': int(id_category_size),
            'JobCodeRecordId': int(id_category_size),
            'ApprovedByManagerUserRecordId': int(id_category_size),
            'IsTimesheet': 4,
            'HoursWorked': 100,
        }

        self.category_fields = {
            **self.initial_category_fields,
            'Time_Event_Month': 14,
            'Time_Event_Day': 33,
            'Time_Event_Hour': 25,
            'Time_Event_Minute': 61,
            'Time_Reference_Month': 14,
            'Time_Reference_Day': 33,
            'Time_Reference_Hour': 25,
            'Time_Reference_Minute': 61
        }

        self.dataset = torch.tensor([
            [1,1,1,1,1,1,0,0,0, 1,29,8,0,0,0,0,0],
            [1,2,1,1,1,1,0,0,0, 1,29,8,15,0,0,0,0],
            [2,1,1,1,1,1,0,0,0, 1,29,16,0,0,0,0,0],
            [2,2,1,1,1,1,0,0,0, 1,29,16,15,0,0,0,0],
            [1,1,1,1,1,1,0,0,0, 1,30,8,0,0,0,0,0],
            [2,1,1,1,1,1,0,0,0, 1,30,16,0,0,0,0,0],
        ])
        self.tgt_dataset = torch.tensor([
            [1,1,1,1,1,1,0,0,0, 2,1,8,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,0, 2,1,8,15,0,0,0,0],
        ])

        self.punch_in_indices   = [0,1,4]
        self.punch_out_indices  = [2,3,5]
        self.recipient_1_indices = [0,2,4,5]
        self.recipient_2_indices = [1,3]

        # for model instantiation
        self.input_size  = len(self.category_fields)
        self.output_size = len(self.category_fields)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        """
        Returns:
          tgt_tensors: List[Tensor(batch=1, tgt_seq, cats)]
          src_streams: List[Dict[str, List[Tensor(batch=1, src_seq, cats)]]]
          masks:       List[List[Tensor(cat_vocab_size)]]
        """
        tgt_tensors = []
        src_streams = []
        masks = []

        for _ in range(self.batch_size):
            streams = {
                'EventType': [
                    self.dataset[self.punch_in_indices, :].unsqueeze(0),
                    self.dataset[self.punch_out_indices, :].unsqueeze(0)
                ],
                'RecipientRecordId': [
                    self.dataset[self.recipient_1_indices, :].unsqueeze(0),
                    self.dataset[self.recipient_2_indices, :].unsqueeze(0)
                ]
            }

            tgt_tensors.append(self.tgt_dataset.unsqueeze(0))  # [1, tgt_seq, cats]

            # prepare masks
            masks.append([
                torch.zeros((self.category_fields[cat],), dtype=torch.float32)
                for cat in self.category_fields
            ])

            src_streams.append(streams)

        return tgt_tensors, src_streams, masks
