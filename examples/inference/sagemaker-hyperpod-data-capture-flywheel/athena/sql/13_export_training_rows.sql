SELECT ticket_text, corrected_category FROM labeled_training_data
            WHERE corrected_category IS NOT NULL ORDER BY ticket_text
