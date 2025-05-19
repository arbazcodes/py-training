# pytorch-model-challenge/data/events_loader.py

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
            'EventType',          # 5 * 1000
            'RecipientRecordId',  # 100 * 1000
        ]

        # 0 is ordinal for all categories to mean "Unknown" or "Not Applicable"
        # All the values going into the model must be ints.
        self.initial_category_fields = {
            'EventType': 10,
            'ActorRecordId': int( id_category_size ),
            'RecipientRecordId': int( id_category_size ),
            'LocationRecordId': int( id_category_size ),
            'CostCodeRecordId': int( id_category_size ),
            'JobCodeRecordId': int( id_category_size ),
            'ApprovedByManagerUserRecordId': int( id_category_size ),
            'IsTimesheet': 4,
            'HoursWorked': 100,
        }

        self.category_fields = {
            **self.initial_category_fields,

            # EventDatetime Conversion
            'Time_Event_Month': 14,
            'Time_Event_Day': 33,
            'Time_Event_Hour': 25,
            'Time_Event_Minute': 61,

            # ReferenceDatetime Conversion
            'Time_Reference_Month': 14,
            'Time_Reference_Day': 33,
            'Time_Reference_Hour': 25,
            'Time_Reference_Minute': 61
        }

        self.dataset = torch.tensor([
            [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 29, 8, 0, 0, 0, 0, 0 ],
            [ 1, 2, 1, 1, 1, 1, 0, 0, 0, 1, 29, 8, 15, 0, 0, 0, 0 ],
            [ 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 29, 16, 0, 0, 0, 0, 0 ],
            [ 2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 29, 16, 15, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 30, 8, 0, 0, 0, 0, 0 ],
            [ 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 30, 16, 0, 0, 0, 0, 0 ],
        ])
        self.tgt_dataset = torch.tensor([
            [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 8, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 8, 15, 0, 0, 0, 0 ],
        ])
        self.punch_in_indices = [0, 1, 4]
        self.punch_out_indices = [ 2, 3, 5]
        self.recipient_1_indices = [0, 2, 4,5]
        self.recipient_2_indices = [ 1, 3 ]

        # expose for train script
        self.input_size = len(self.category_fields)
        self.output_size = len(self.category_fields)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        tgt_tensors = []
        src_streams = []
        masks = []
        for i in range(self.batch_size):
            src_streams.append(
                {
                    'EventType': [
                        torch.tensor(self.dataset[self.punch_in_indices, :]),
                        torch.tensor(self.dataset[self.punch_out_indices, :])
                    ],
                    'RecipientRecordId': [
                        torch.tensor(self.dataset[self.recipient_1_indices, :]),
                        torch.tensor(self.dataset[self.recipient_2_indices, :])
                    ]
                }
            )
            tgt_tensors.append(self.tgt_dataset)
            masks.append(
                [ torch.zeros((self.category_fields[category],)) for category in self.category_fields.keys() ]
            )

        return (
            tgt_tensors,   
            src_streams,   
            masks
        )
