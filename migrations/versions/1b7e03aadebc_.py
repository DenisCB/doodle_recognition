"""empty message

Revision ID: 1b7e03aadebc
Revises: 
Create Date: 2019-12-24 16:03:23.841935

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1b7e03aadebc'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('drawings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('predicted_label', sa.String(length=100), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('drawings')
    # ### end Alembic commands ###